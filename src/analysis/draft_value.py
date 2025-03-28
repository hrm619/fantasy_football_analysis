import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_adp_vs_performance(
    df: pd.DataFrame, team_count: int = 12
) -> Dict[str, pd.DataFrame]:
    """
    Analyze ADP vs. performance by draft round and position.

    Args:
        df: Player performance dataframe with ADP
        team_count: Number of teams in the league

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of dataframes with ADP vs. performance analysis
    """
    logger.info("Analyzing ADP vs. performance")

    # Check for required columns
    required_cols = ["Player", "FantPos", "ADP", "Half_PPR", "Half_PPR_PPG"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return {}

    # Drop rows with missing ADP
    analysis_df = df.dropna(subset=["ADP"]).copy()

    # Calculate draft round
    analysis_df["Draft_Round"] = np.ceil(analysis_df["ADP"] / team_count).astype(int)

    # Limit to reasonable number of rounds (players actually drafted)
    max_round = min(20, analysis_df["Draft_Round"].max())
    analysis_df = analysis_df[analysis_df["Draft_Round"] <= max_round]

    results = {}

    # Calculate overall round stats
    round_stats = []
    for round_num in range(1, max_round + 1):
        round_df = analysis_df[analysis_df["Draft_Round"] == round_num]
        if not round_df.empty:
            # Calculate expected points for round based on ADP order
            round_df = round_df.sort_values("ADP").reset_index(drop=True)

            # Define success/bust thresholds: success = top 25% of actual scores, bust = bottom 25%
            all_scores = analysis_df["Half_PPR"]
            success_threshold = all_scores.quantile(0.75)
            bust_threshold = all_scores.quantile(0.25)

            stat = {
                "Draft_Round": round_num,
                "Player_Count": len(round_df),
                "Avg_Points": round_df["Half_PPR"].mean(),
                "Std_Dev": round_df["Half_PPR"].std(),
                "Avg_PPG": round_df["Half_PPR_PPG"].mean(),
                "Success_Rate": (round_df["Half_PPR"] >= success_threshold).mean()
                * 100,
                "Bust_Rate": (round_df["Half_PPR"] <= bust_threshold).mean() * 100,
                "Top_Player": (
                    round_df.loc[round_df["Half_PPR"].idxmax(), "Player"]
                    if not round_df.empty
                    else ""
                ),
                "Top_Points": round_df["Half_PPR"].max(),
                "Worst_Player": (
                    round_df.loc[round_df["Half_PPR"].idxmin(), "Player"]
                    if not round_df.empty
                    else ""
                ),
                "Worst_Points": round_df["Half_PPR"].min(),
            }
            round_stats.append(stat)

    results["round_stats"] = pd.DataFrame(round_stats)

    # Calculate position-specific round stats
    positions = ["QB", "RB", "WR", "TE"]
    position_round_stats = []

    for pos in positions:
        pos_df = analysis_df[analysis_df["FantPos"] == pos]
        for round_num in range(1, max_round + 1):
            pos_round_df = pos_df[pos_df["Draft_Round"] == round_num]
            if (
                len(pos_round_df) >= 2
            ):  # Require at least 2 players for meaningful stats
                all_pos_scores = analysis_df[analysis_df["FantPos"] == pos]["Half_PPR"]
                success_threshold = all_pos_scores.quantile(0.75)
                bust_threshold = all_pos_scores.quantile(0.25)

                stat = {
                    "Position": pos,
                    "Draft_Round": round_num,
                    "Player_Count": len(pos_round_df),
                    "Avg_Points": pos_round_df["Half_PPR"].mean(),
                    "Std_Dev": pos_round_df["Half_PPR"].std(),
                    "Avg_PPG": pos_round_df["Half_PPR_PPG"].mean(),
                    "Success_Rate": (
                        pos_round_df["Half_PPR"] >= success_threshold
                    ).mean()
                    * 100,
                    "Bust_Rate": (pos_round_df["Half_PPR"] <= bust_threshold).mean()
                    * 100,
                }
                position_round_stats.append(stat)

    results["position_round_stats"] = pd.DataFrame(position_round_stats)

    # Calculate optimal draft slots by position
    optimal_positions = []

    for round_num in range(1, max_round + 1):
        round_df = analysis_df[analysis_df["Draft_Round"] == round_num]
        if not round_df.empty:
            # Group by position and calculate average points
            pos_stats = (
                round_df.groupby("FantPos")["Half_PPR"]
                .agg(["mean", "count"])
                .reset_index()
            )
            pos_stats = pos_stats[pos_stats["count"] >= 2]  # Require at least 2 players

            if not pos_stats.empty:
                # Find best position for this round
                best_pos = pos_stats.loc[pos_stats["mean"].idxmax()]

                optimal_positions.append(
                    {
                        "Draft_Round": round_num,
                        "Optimal_Position": best_pos["FantPos"],
                        "Avg_Points": best_pos["mean"],
                        "Player_Count": best_pos["count"],
                    }
                )

    results["optimal_positions_by_round"] = pd.DataFrame(optimal_positions)

    logger.info("ADP vs. performance analysis completed")
    return results


def calculate_vbd_rankings(
    df: pd.DataFrame, baseline_values: Dict[str, int] = None
) -> pd.DataFrame:
    """
    Calculate Value Based Drafting (VBD) rankings for all players.

    Args:
        df: Player performance dataframe
        baseline_values: Dictionary mapping positions to baseline ranks

    Returns:
        pd.DataFrame: Dataframe with VBD rankings
    """
    logger.info("Calculating VBD rankings")

    # Set default baseline values if not provided
    if baseline_values is None:
        baseline_values = {"QB": 12, "RB": 24, "WR": 24, "TE": 12}

    result_df = df.copy()

    # Check if VORP is already calculated
    if "VORP" not in result_df.columns:
        logger.warning("VORP not found in dataframe, calculating now")

        # Define replacement level values
        replacement_values = {}

        for pos, baseline_rank in baseline_values.items():
            pos_df = result_df[result_df["FantPos"] == pos].copy()

            if len(pos_df) >= baseline_rank:
                # Sort by half PPR points
                pos_df = pos_df.sort_values("Half_PPR", ascending=False).reset_index(
                    drop=True
                )

                # Get replacement level points
                replacement_values[pos] = pos_df.iloc[baseline_rank - 1]["Half_PPR"]
            else:
                logger.warning(
                    f"Not enough {pos} players for baseline rank {baseline_rank}"
                )
                replacement_values[pos] = 0

        # Calculate VBD for each position
        for pos, replacement_value in replacement_values.items():
            pos_mask = result_df["FantPos"] == pos
            result_df.loc[pos_mask, "VBD"] = (
                result_df.loc[pos_mask, "Half_PPR"] - replacement_value
            )
    else:
        # Use existing VORP as VBD
        result_df["VBD"] = result_df["VORP"]

    # Set VBD to 0 for players below replacement level
    result_df["VBD"] = result_df["VBD"].clip(lower=0)

    # Rank players by VBD
    result_df["VBD_Rank"] = result_df["VBD"].rank(ascending=False, method="min")

    # Calculate VBD by ADP
    if "ADP" in result_df.columns:
        # Find VBD per ADP point
        result_df["VBD_per_ADP"] = result_df["VBD"] / result_df["ADP"]

        # Replace inf/NaN values
        result_df["VBD_per_ADP"] = (
            result_df["VBD_per_ADP"].replace([np.inf, -np.inf], np.nan).fillna(0)
        )

        # Calculate draft round
        team_count = 12  # Default to 12-team league
        result_df["Draft_Round"] = np.ceil(result_df["ADP"] / team_count).astype(int)

        # Calculate VBD by round
        round_vbd = []
        max_round = (
            min(20, result_df["Draft_Round"].max())
            if "Draft_Round" in result_df.columns
            else 15
        )

        for round_num in range(1, max_round + 1):
            if "Draft_Round" in result_df.columns:
                round_df = result_df[result_df["Draft_Round"] == round_num]
                if not round_df.empty:
                    for pos in ["QB", "RB", "WR", "TE"]:
                        pos_round_df = round_df[round_df["FantPos"] == pos]
                        if len(pos_round_df) >= 2:
                            round_vbd.append(
                                {
                                    "Draft_Round": round_num,
                                    "Position": pos,
                                    "Avg_VBD": pos_round_df["VBD"].mean(),
                                    "Avg_VBD_per_ADP": pos_round_df[
                                        "VBD_per_ADP"
                                    ].mean(),
                                    "Player_Count": len(pos_round_df),
                                }
                            )

        if round_vbd:
            result_df.attrs["vbd_by_round"] = pd.DataFrame(round_vbd)

    logger.info("VBD ranking calculation completed")
    return result_df
