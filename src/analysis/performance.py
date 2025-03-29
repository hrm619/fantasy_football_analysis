import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_performance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate standard performance metrics for all players.

    Args:
        df: Master player dataframe

    Returns:
        pd.DataFrame: Dataframe with performance metrics added
    """
    logger.info("Calculating performance metrics")

    # Ensure required columns exist
    required_cols = ["Player", "FantPos", "G", "Half_PPR", "Half_PPR_PPG"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return df

    # Calculate additional metrics
    result_df = df.copy()

    # Create position-specific subsets
    positions = ["QB", "RB", "WR", "TE"]
    pos_dfs = {pos: result_df[result_df["FantPos"] == pos].copy() for pos in positions}

    # Calculate positional ranks based on total points
    for pos, pos_df in pos_dfs.items():
        pos_df = pos_df.sort_values("Half_PPR", ascending=False).reset_index(drop=True)
        pos_df[f"{pos}_Rank"] = pos_df.index + 1
        result_df = pd.merge(
            result_df,
            pos_df[["Player", "Team_std", f"{pos}_Rank"]],
            on=["Player", "Team_std"],
            how="left",
        )

    # Create FLX rank (WR/RB/TE)
    flx_df = pd.concat([pos_dfs["WR"], pos_dfs["RB"], pos_dfs["TE"]])
    flx_df = flx_df.sort_values("Half_PPR", ascending=False).reset_index(drop=True)
    flx_df["FLX_Rank"] = flx_df.index + 1
    result_df = pd.merge(
        result_df,
        flx_df[["Player", "Team_std", "FLX_Rank"]],
        on=["Player", "Team_std"],
        how="left",
    )

    # Calculate overall rank
    result_df["Overall_Rank"] = result_df["Half_PPR"].rank(
        ascending=False, method="min"
    )

    # Calculate position-specific metrics
    # For QBs
    qb_metrics = calculate_qb_metrics(result_df[result_df["FantPos"] == "QB"])
    result_df = pd.merge(result_df, qb_metrics, on=["Player", "Team_std"], how="left")

    # For RBs
    rb_metrics = calculate_rb_metrics(result_df[result_df["FantPos"] == "RB"])
    result_df = pd.merge(result_df, rb_metrics, on=["Player", "Team_std"], how="left")

    # For WRs
    wr_metrics = calculate_wr_metrics(result_df[result_df["FantPos"] == "WR"])
    result_df = pd.merge(result_df, wr_metrics, on=["Player", "Team_std"], how="left")

    # For TEs
    te_metrics = calculate_te_metrics(result_df[result_df["FantPos"] == "TE"])
    result_df = pd.merge(result_df, te_metrics, on=["Player", "Team_std"], how="left")

    logger.info("Performance metrics calculation completed")
    return result_df


def calculate_qb_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate QB-specific performance metrics.

    Args:
        df: QB player dataframe

    Returns:
        pd.DataFrame: Dataframe with QB metrics
    """
    logger.info("Calculating QB-specific metrics")

    if df.empty:
        logger.warning("Empty QB dataframe provided")
        return pd.DataFrame()

    result_df = df[["Player", "Team_std"]].copy()

    # Calculate passing points share
    if all(col in df.columns for col in ["Passing TD", "Passing Yds", "Half_PPR"]):
        result_df["Pass_Points"] = (
            (df["Passing Yds"] / 25) + (df["Passing TD"] * 4) - (df["Passing Int"] * 2)
        )
        result_df["Pass_Points_Share"] = result_df["Pass_Points"] / df["Half_PPR"]

    # Calculate rushing points share
    if all(col in df.columns for col in ["Rushing TD", "Rushing Yds", "Half_PPR"]):
        result_df["Rush_Points"] = (df["Rushing Yds"] / 10) + (df["Rushing TD"] * 6)
        result_df["Rush_Points_Share"] = result_df["Rush_Points"] / df["Half_PPR"]

    # Calculate efficiency metrics
    if all(col in df.columns for col in ["Passing Att", "Passing Yds", "Passing TD"]):
        result_df["Pass_Yards_Per_Att"] = df["Passing Yds"] / df["Passing Att"].replace(
            0, np.nan
        )
        result_df["TD_Rate"] = (
            df["Passing TD"] / df["Passing Att"].replace(0, np.nan) * 100
        )

    if all(col in df.columns for col in ["Passing Att", "Passing Int"]):
        result_df["Int_Rate"] = (
            df["Passing Int"] / df["Passing Att"].replace(0, np.nan) * 100
        )

    # Replace NaN values
    result_df = result_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info("QB metrics calculation completed")
    return result_df


def calculate_rb_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate RB-specific performance metrics.

    Args:
        df: RB player dataframe

    Returns:
        pd.DataFrame: Dataframe with RB metrics
    """
    logger.info("Calculating RB-specific metrics")

    if df.empty:
        logger.warning("Empty RB dataframe provided")
        return pd.DataFrame()

    result_df = df[["Player", "Team_std"]].copy()

    # Calculate rushing points share
    if all(col in df.columns for col in ["Rushing TD", "Rushing Yds", "Half_PPR"]):
        result_df["Rush_Points"] = (df["Rushing Yds"] / 10) + (df["Rushing TD"] * 6)
        result_df["Rush_Points_Share"] = result_df["Rush_Points"] / df[
            "Half_PPR"
        ].replace(0, np.nan)

    # Calculate receiving points share
    if all(
        col in df.columns
        for col in ["Receiving Rec", "Receiving Yds", "Receiving TD", "Half_PPR"]
    ):
        result_df["Recv_Points"] = (
            (df["Receiving Rec"] * 0.5)
            + (df["Receiving Yds"] / 10)
            + (df["Receiving TD"] * 6)
        )
        result_df["Recv_Points_Share"] = result_df["Recv_Points"] / df[
            "Half_PPR"
        ].replace(0, np.nan)

    # Calculate efficiency metrics
    if all(col in df.columns for col in ["Rushing Att", "Rushing Yds"]):
        result_df["Rush_Yards_Per_Att"] = df["Rushing Yds"] / df["Rushing Att"].replace(
            0, np.nan
        )

    if all(col in df.columns for col in ["Receiving Tgt", "Receiving Rec"]):
        result_df["Catch_Rate"] = df["Receiving Rec"] / df["Receiving Tgt"].replace(
            0, np.nan
        )

    # Replace NaN values
    result_df = result_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info("RB metrics calculation completed")
    return result_df


def calculate_wr_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate WR-specific performance metrics.

    Args:
        df: WR player dataframe

    Returns:
        pd.DataFrame: Dataframe with WR metrics
    """
    logger.info("Calculating WR-specific metrics")

    if df.empty:
        logger.warning("Empty WR dataframe provided")
        return pd.DataFrame()

    result_df = df[["Player", "Team_std"]].copy()

    # Calculate receiving points
    if all(
        col in df.columns for col in ["Receiving Rec", "Receiving Yds", "Receiving TD"]
    ):
        result_df["Recv_Points"] = (
            (df["Receiving Rec"] * 0.5)
            + (df["Receiving Yds"] / 10)
            + (df["Receiving TD"] * 6)
        )

    # Calculate efficiency metrics
    if all(col in df.columns for col in ["Receiving Tgt", "Receiving Rec"]):
        result_df["Catch_Rate"] = df["Receiving Rec"] / df["Receiving Tgt"].replace(
            0, np.nan
        )

    if all(col in df.columns for col in ["Receiving Yds", "Receiving Rec"]):
        result_df["Yards_Per_Reception"] = df["Receiving Yds"] / df[
            "Receiving Rec"
        ].replace(0, np.nan)

    if all(col in df.columns for col in ["Receiving TD", "Receiving Rec"]):
        result_df["TD_Rate"] = (
            df["Receiving TD"] / df["Receiving Rec"].replace(0, np.nan) * 100
        )

    # Check if advanced receiving metrics are available
    if "yprr" in df.columns:
        result_df["YPRR"] = df["yprr"]

    if "avg_depth_of_target" in df.columns:
        result_df["aDOT"] = df["avg_depth_of_target"]

    # Replace NaN values
    result_df = result_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info("WR metrics calculation completed")
    return result_df


def calculate_te_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate TE-specific performance metrics.

    Args:
        df: TE player dataframe

    Returns:
        pd.DataFrame: Dataframe with TE metrics
    """
    logger.info("Calculating TE-specific metrics")

    if df.empty:
        logger.warning("Empty TE dataframe provided")
        return pd.DataFrame()

    # TE metrics are similar to WR metrics
    result_df = calculate_wr_metrics(df)

    # Additional TE-specific metrics
    if "route_rate" in df.columns:
        result_df["Route_Rate"] = df["route_rate"]

    if "inline_rate" in df.columns:
        result_df["Inline_Rate"] = df["inline_rate"]

    logger.info("TE metrics calculation completed")
    return result_df


def analyze_expectation_vs_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the gap between preseason expectations and actual performance.

    Args:
        df: Master player dataframe with preseason rankings

    Returns:
        pd.DataFrame: Dataframe with expectation vs. performance analysis
    """
    logger.info("Analyzing expectation vs. performance")

    # Check for required columns
    required_cols = ["Player", "FantPos", "ADP", "Half_PPR", "Half_PPR_PPG"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return df

    result_df = df.copy()

    # Calculate ADP rank vs actual rank delta
    if "ADP" in result_df.columns and "Overall_Rank" in result_df.columns:
        # Convert ADP to rank (lower ADP = better rank)
        result_df["ADP_Rank"] = result_df["ADP"].rank(method="min")
        result_df["ADP_vs_Actual_Rank_Delta"] = (
            result_df["ADP_Rank"] - result_df["Overall_Rank"]
        )

    # Calculate projected vs actual points delta
    proj_points_cols = [
        col
        for col in result_df.columns
        if "Points" in col
        and col
        not in ["Half_PPR", "Points", "Pass_Points", "Rush_Points", "Recv_Points"]
    ]

    if proj_points_cols and "Half_PPR" in result_df.columns:
        for col in proj_points_cols:
            delta_col = f"{col}_vs_Actual_Delta"
            pct_col = f"{col}_vs_Actual_Pct"

            # Calculate absolute and percentage difference
            result_df[delta_col] = result_df["Half_PPR"] - result_df[col]
            result_df[pct_col] = (
                result_df["Half_PPR"] / result_df[col].replace(0, np.nan) - 1
            ) * 100

            # Replace NaN values
            result_df[pct_col] = (
                result_df[pct_col].replace([np.inf, -np.inf], np.nan).fillna(0)
            )

    # Calculate positional rank deltas
    pos_rank_cols = [col for col in result_df.columns if "Pos Rank" in col]
    actual_rank_cols = {
        "QB": "QB_Rank",
        "RB": "RB_Rank",
        "WR": "WR_Rank",
        "TE": "TE_Rank",
    }

    for pos, actual_col in actual_rank_cols.items():
        for proj_col in pos_rank_cols:
            if actual_col in result_df.columns:
                pos_subset = result_df[result_df["FantPos"] == pos]
                if not pos_subset.empty:
                    # Create new column name for delta
                    delta_col = f"{proj_col}_vs_Actual_Delta"

                    # Convert to string and handle NaN values before extracting numeric rank
                    result_df[f"{proj_col}_Numeric"] = (
                        result_df[proj_col]
                        .fillna("")
                        .astype(str)
                        .str.extract(r"(\d+)")
                        .astype(float)
                        .fillna(np.nan)
                    )

                    # Calculate delta
                    result_df[delta_col] = (
                        result_df[f"{proj_col}_Numeric"] - result_df[actual_col]
                    )

    # Categorize performance relative to expectations
    if "PFF Points_vs_Actual_Pct" in result_df.columns:
        # Use PFF projections for categorization
        result_df["Performance_Category"] = pd.cut(
            result_df["PFF Points_vs_Actual_Pct"],
            bins=[-float("inf"), -20, -5, 5, 20, float("inf")],
            labels=[
                "Significant Underperformer",
                "Moderate Underperformer",
                "Met Expectations",
                "Moderate Overperformer",
                "Significant Overperformer",
            ],
        )

    logger.info("Expectation vs. performance analysis completed")
    return result_df
