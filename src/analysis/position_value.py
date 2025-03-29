import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_vorp(df: pd.DataFrame, baselines: Dict[str, int]) -> pd.DataFrame:
    """
    Calculate Value Over Replacement Player (VORP) for each player.

    Args:
        df: Player performance dataframe
        baselines: Dictionary mapping positions to baseline ranks

    Returns:
        pd.DataFrame: Dataframe with VORP metrics added
    """
    logger.info("Calculating Value Over Replacement Player (VORP)")

    result_df = df.copy()

    # Define replacement level values
    replacement_values = {}

    for pos, baseline_rank in baselines.items():
        pos_df = result_df[result_df["FantPos"] == pos].copy()
        
        # Log position stats
        logger.info(f"Processing {pos}: {len(pos_df)} players found")

        if len(pos_df) >= baseline_rank:
            # Sort by half PPR points
            pos_df = pos_df.sort_values("Half_PPR", ascending=False).reset_index(drop=True)

            # Get replacement level points (player at baseline rank)
            try:
                replacement_values[pos] = pos_df.iloc[baseline_rank - 1]["Half_PPR"]
                logger.info(
                    f"{pos} replacement level ({pos}{baseline_rank}): {replacement_values[pos]:.2f} points"
                )
            except IndexError:
                logger.warning(f"Could not find {pos}{baseline_rank}, using last available player")
                replacement_values[pos] = pos_df.iloc[-1]["Half_PPR"]
        else:
            logger.warning(
                f"Not enough {pos} players to establish baseline at rank {baseline_rank}"
            )
            replacement_values[pos] = 0

    # Calculate VORP for each position
    for pos, replacement_value in replacement_values.items():
        pos_mask = result_df["FantPos"] == pos
        result_df.loc[pos_mask, "VORP"] = (
            result_df.loc[pos_mask, "Half_PPR"] - replacement_value
        )

        # Calculate VORP per game
        if "G" in result_df.columns:
            result_df.loc[pos_mask, "VORP_Per_Game"] = (
                result_df.loc[pos_mask, "VORP"] / result_df.loc[pos_mask, "G"]
            )

    # Set VORP to 0 for players below replacement level
    result_df["VORP"] = result_df["VORP"].clip(lower=0)
    result_df["VORP_Per_Game"] = result_df["VORP_Per_Game"].clip(lower=0)

    # Rank players by VORP
    result_df["VORP_Rank"] = result_df["VORP"].rank(ascending=False, method="min")

    logger.info("VORP calculation completed")
    return result_df


def analyze_positional_scarcity(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze positional scarcity by calculating point differentials within positions.

    Args:
        df: Player performance dataframe with ranks

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of dataframes with scarcity analysis by position
    """
    logger.info("Analyzing positional scarcity")

    positions = ["QB", "RB", "WR", "TE"]
    tiers = {
        "QB": [(1, 6), (7, 12), (13, 18)],
        "RB": [(1, 12), (13, 24), (25, 36)],
        "WR": [(1, 12), (13, 24), (25, 36)],
        "TE": [(1, 6), (7, 12), (13, 18)],
    }

    results = {}

    for pos in positions:
        # Filter players by position and sort by points
        pos_df = df[df["FantPos"] == pos].copy()
        pos_df = pos_df.sort_values("Half_PPR", ascending=False).reset_index(drop=True)
        
        # Log position stats
        logger.info(f"Processing {pos}: {len(pos_df)} players found")

        # Calculate tier stats
        tier_results = []
        for start_rank, end_rank in tiers[pos]:
            if len(pos_df) >= end_rank:
                try:
                    tier_data = pos_df.iloc[start_rank - 1 : end_rank].copy()

                    # Calculate statistics
                    tier_stats = {
                        "Position": pos,
                        "Tier": f"{pos}{start_rank}-{pos}{end_rank}",
                        "Avg_Points": tier_data["Half_PPR"].mean(),
                        "Std_Dev": tier_data["Half_PPR"].std(),
                        "Max_Points": tier_data["Half_PPR"].max(),
                        "Min_Points": tier_data["Half_PPR"].min(),
                        "Range": tier_data["Half_PPR"].max() - tier_data["Half_PPR"].min(),
                    }

                    # Calculate point differential to next tier if possible
                    if end_rank < len(pos_df) and start_rank > 1:
                        next_tier_avg = pos_df.iloc[end_rank : end_rank + 5]["Half_PPR"].mean()
                        prev_tier_avg = pos_df.iloc[start_rank - 6 : start_rank - 1]["Half_PPR"].mean()
                        tier_stats["Next_Tier_Diff"] = tier_stats["Avg_Points"] - next_tier_avg
                        tier_stats["Prev_Tier_Diff"] = prev_tier_avg - tier_stats["Avg_Points"]

                    tier_results.append(tier_stats)
                except IndexError:
                    logger.warning(f"Could not process tier {pos}{start_rank}-{pos}{end_rank}")
                    continue
            else:
                logger.warning(f"Not enough {pos} players for tier {start_rank}-{end_rank}")

        # Calculate point differentials between adjacent ranks
        rank_diffs = []
        for i in range(len(pos_df) - 1):
            try:
                diff = {
                    "Rank_Higher": i + 1,
                    "Rank_Lower": i + 2,
                    "Player_Higher": pos_df.iloc[i]["Player"],
                    "Player_Lower": pos_df.iloc[i + 1]["Player"],
                    "Points_Higher": pos_df.iloc[i]["Half_PPR"],
                    "Points_Lower": pos_df.iloc[i + 1]["Half_PPR"],
                    "Point_Diff": pos_df.iloc[i]["Half_PPR"] - pos_df.iloc[i + 1]["Half_PPR"],
                    "Pct_Diff": (
                        (pos_df.iloc[i]["Half_PPR"] / pos_df.iloc[i + 1]["Half_PPR"] - 1) * 100
                        if pos_df.iloc[i + 1]["Half_PPR"] > 0
                        else 0
                    ),
                }
                rank_diffs.append(diff)
            except IndexError:
                logger.warning(f"Could not calculate rank differential for {pos} rank {i+1}")
                continue

        # Create results dataframes
        if tier_results:
            results[f"{pos}_tiers"] = pd.DataFrame(tier_results)
        if rank_diffs:
            results[f"{pos}_rank_diffs"] = pd.DataFrame(rank_diffs)

    # Calculate positional value adjustment factors
    pos_value_adjustments = calculate_position_value_adjustments(df)
    results["position_value_adjustments"] = pos_value_adjustments

    logger.info("Positional scarcity analysis completed")
    return results


def calculate_position_value_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate position value adjustment factors based on scarcity.

    Args:
        df: Player performance dataframe

    Returns:
        pd.DataFrame: Dataframe with position value adjustment factors
    """
    logger.info("Calculating position value adjustment factors")

    positions = ["QB", "RB", "WR", "TE"]
    results = []

    for pos in positions:
        pos_df = df[df["FantPos"] == pos].copy()
        if len(pos_df) > 0:
            pos_df = pos_df.sort_values("Half_PPR", ascending=False).reset_index(drop=True)
            
            # Log position stats
            logger.info(f"Processing {pos}: {len(pos_df)} players found")

            # Calculate stats for startable players
            start_counts = {"QB": 12, "RB": 24, "WR": 24, "TE": 12}
            start_count = start_counts.get(pos, 12)

            if len(pos_df) >= start_count:
                try:
                    startable_df = pos_df.iloc[:start_count]
                    
                    # Get top score safely
                    top_score = pos_df.iloc[0]["Half_PPR"] if len(pos_df) > 0 else 0
                    
                    # Get starter cutoff safely
                    starter_cutoff = (
                        pos_df.iloc[start_count - 1]["Half_PPR"]
                        if len(pos_df) >= start_count
                        else 0
                    )
                    
                    # Calculate top to starter difference safely
                    top_to_starter_diff = (
                        top_score - starter_cutoff
                        if len(pos_df) >= start_count
                        else 0
                    )

                    pos_stats = {
                        "Position": pos,
                        "Startable_Count": start_count,
                        "Avg_Startable_Points": startable_df["Half_PPR"].mean(),
                        "Std_Dev_Startable": startable_df["Half_PPR"].std(),
                        "Top_Score": top_score,
                        "Starter_Cutoff": starter_cutoff,
                        "Top_To_Starter_Diff": top_to_starter_diff,
                    }

                    results.append(pos_stats)
                except IndexError as e:
                    logger.warning(f"Error processing {pos} stats: {str(e)}")
                    continue
            else:
                logger.warning(f"Not enough {pos} players for startable analysis (need {start_count}, found {len(pos_df)})")

    # Create dataframe and calculate adjustment factors
    result_df = pd.DataFrame(results)

    if not result_df.empty:
        try:
            # Normalize standard deviations to calculate adjustment factors
            # Higher std dev means more positional advantage
            total_std = result_df["Std_Dev_Startable"].sum()
            if total_std > 0:  # Prevent division by zero
                result_df["Value_Adjustment_Factor"] = (
                    result_df["Std_Dev_Startable"] / total_std * len(positions)
                )
            else:
                logger.warning("Total standard deviation is 0, setting all adjustment factors to 1")
                result_df["Value_Adjustment_Factor"] = 1.0

            # Calculate adjusted points for each position
            for pos in positions:
                try:
                    adjustment = (
                        result_df.loc[
                            result_df["Position"] == pos, "Value_Adjustment_Factor"
                        ].values[0]
                        if pos in result_df["Position"].values
                        else 1.0
                    )
                    pos_mask = df["FantPos"] == pos
                    df.loc[pos_mask, "Position_Adjusted_Points"] = (
                        df.loc[pos_mask, "Half_PPR"] * adjustment
                    )
                except IndexError as e:
                    logger.warning(f"Error calculating adjusted points for {pos}: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error calculating adjustment factors: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error

    logger.info("Position value adjustment calculation completed")
    return result_df


def analyze_position_groups(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze position groups (RB/WR vs. QB/TE) to optimize draft capital allocation.

    Args:
        df: Player performance dataframe

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of dataframes with position group analysis
    """
    logger.info("Analyzing position groups")

    # Define position groups
    position_groups = {"Primary_Skill": ["RB", "WR"], "Onesies": ["QB", "TE"]}

    results = {}

    # Calculate stats for each group
    for group_name, positions in position_groups.items():
        group_df = df[df["FantPos"].isin(positions)].copy()
        
        # Log group stats
        logger.info(f"Processing {group_name}: {len(group_df)} players found")

        # Create stats by position within group
        pos_stats = []
        for pos in positions:
            pos_df = group_df[group_df["FantPos"] == pos].copy()
            if not pos_df.empty:
                pos_df = pos_df.sort_values("Half_PPR", ascending=False).reset_index(drop=True)
                
                # Log position stats
                logger.info(f"  {pos}: {len(pos_df)} players found")

                # Define tiers based on position
                tier_sizes = {"QB": 6, "RB": 12, "WR": 12, "TE": 6}
                tier_size = tier_sizes.get(pos, 12)

                # Calculate stats for top tier
                if len(pos_df) >= tier_size:
                    try:
                        top_tier = pos_df.iloc[:tier_size]
                        
                        # Get top score safely
                        top_score = pos_df.iloc[0]["Half_PPR"] if len(pos_df) > 0 else 0
                        
                        # Get tier end score safely
                        tier_end_score = (
                            pos_df.iloc[tier_size - 1]["Half_PPR"]
                            if len(pos_df) >= tier_size
                            else 0
                        )

                        stat = {
                            "Position": pos,
                            "Group": group_name,
                            "Total_Points": pos_df["Half_PPR"].sum(),
                            "Avg_Points": pos_df["Half_PPR"].mean(),
                            "Top_Tier_Avg": top_tier["Half_PPR"].mean(),
                            "Top_Tier_Std": top_tier["Half_PPR"].std(),
                            "Top_Score": top_score,
                            "Top_To_Tier_End_Diff": top_score - tier_end_score,
                        }

                        # Add ADP stats if available
                        if "ADP" in pos_df.columns:
                            stat["Avg_ADP"] = pos_df["ADP"].mean()
                            stat["Top_Tier_Avg_ADP"] = top_tier["ADP"].mean()

                        pos_stats.append(stat)
                    except IndexError as e:
                        logger.warning(f"Error processing {pos} stats: {str(e)}")
                        continue
                else:
                    logger.warning(f"Not enough {pos} players for tier analysis (need {tier_size}, found {len(pos_df)})")

        # Calculate overall group stats
        if group_df.empty:
            logger.warning(f"Empty dataframe for position group {group_name}")
            continue

        try:
            group_stats = {
                "Group": group_name,
                "Total_Points": group_df["Half_PPR"].sum(),
                "Avg_Points": group_df["Half_PPR"].mean(),
                "Std_Dev": group_df["Half_PPR"].std(),
                "Player_Count": len(group_df),
            }

            # Add ADP stats if available
            if "ADP" in group_df.columns:
                group_stats["Avg_ADP"] = group_df["ADP"].mean()
                group_stats["ADP_Std"] = group_df["ADP"].std()

            # Store results
            if pos_stats:
                results[f"{group_name}_position_stats"] = pd.DataFrame(pos_stats)
            results[f"{group_name}_group_stats"] = pd.DataFrame([group_stats])
        except Exception as e:
            logger.error(f"Error calculating group stats for {group_name}: {str(e)}")
            continue

    # Calculate optimal draft capital allocation based on production
    if all(f"{group}_group_stats" in results for group in position_groups.keys()):
        try:
            primary_total = results["Primary_Skill_group_stats"]["Total_Points"].values[0]
            onesies_total = results["Onesies_group_stats"]["Total_Points"].values[0]
            total_points = primary_total + onesies_total

            if total_points > 0:  # Prevent division by zero
                primary_alloc = primary_total / total_points
                onesies_alloc = onesies_total / total_points

                # Create allocation dataframe
                allocation_data = [
                    {
                        "Group": "Primary_Skill",
                        "Positions": "RB/WR",
                        "Point_Share": primary_total / total_points * 100,
                        "Recommended_Draft_Capital": primary_alloc * 100,
                    },
                    {
                        "Group": "Onesies",
                        "Positions": "QB/TE",
                        "Point_Share": onesies_total / total_points * 100,
                        "Recommended_Draft_Capital": onesies_alloc * 100,
                    },
                ]
                results["draft_capital_allocation"] = pd.DataFrame(allocation_data)
            else:
                logger.warning("Total points is 0, skipping draft capital allocation")
        except Exception as e:
            logger.error(f"Error calculating draft capital allocation: {str(e)}")

    logger.info("Position group analysis completed")
    return results
