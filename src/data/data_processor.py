import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from .team_standardizer import standardize_team_names_across_datasets
from src.utils.validation import validate_player_performance_data

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def standardize_team_names(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Wrapper function that uses the team_standardizer module to standardize team names.
    
    Args:
        data_dict: Dictionary of dataframes to process
        
    Returns:
        Dict[str, pd.DataFrame]: Processed dataframes with standardized team names
    """
    return standardize_team_names_across_datasets(data_dict)


def filter_season_data(
    data_dict: Dict[str, pd.DataFrame], season: int = 2024
) -> Dict[str, pd.DataFrame]:
    """
    Filter all dataframes to include only the specified season.

    Args:
        data_dict: Dictionary of dataframes to filter
        season: Season year to filter on

    Returns:
        Dict[str, pd.DataFrame]: Filtered dataframes
    """
    logger.info(f"Filtering data for season {season}")

    filtered_dict = {}
    for key, df in data_dict.items():
        if "Season" in df.columns:
            filtered_dict[key] = df[df["Season"] == season].copy()
            logger.info(
                f"Filtered {key}: {len(df)} rows → {len(filtered_dict[key])} rows"
            )
        elif "season" in df.columns:
            filtered_dict[key] = df[df["season"] == season].copy()
            logger.info(
                f"Filtered {key}: {len(df)} rows → {len(filtered_dict[key])} rows"
            )
        else:
            filtered_dict[key] = df.copy()
            logger.warning(f"No season column found in {key}, using all data")

    return filtered_dict


def create_master_player_dataset(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a master dataset with player data from all sources.

    Args:
        data_dict: Dictionary of processed dataframes

    Returns:
        pd.DataFrame: Master player dataset
    """
    logger.info("Creating master player dataset")

    # Start with season data as the base
    if "season_data" not in data_dict:
        logger.error("Season data not found in data dictionary")
        return pd.DataFrame()

    master_df = data_dict["season_data"].copy()

    # Merge with preseason rankings
    if "preseason_rankings" in data_dict:
        preseason_df = data_dict["preseason_rankings"].copy()
        master_df = pd.merge(
            master_df,
            preseason_df,
            on=["Player", "Team_std"],
            how="left",
            suffixes=("", "_preseason"),
        )
        logger.info(f"Merged preseason rankings: {master_df.shape}")

    # Add position-specific data
    # For QBs
    if "passing_data" in data_dict:
        qb_df = data_dict["passing_data"].copy()
        master_df = pd.merge(
            master_df,
            qb_df,
            left_on=["Player", "Team_std"],
            right_on=["player", "Team_std"],
            how="left",
            suffixes=("", "_passing"),
        )
        logger.info(f"Merged passing data: {master_df.shape}")

    # For receivers
    if "receiving_data" in data_dict:
        rec_df = data_dict["receiving_data"].copy()
        master_df = pd.merge(
            master_df,
            rec_df,
            left_on=["Player", "Team_std"],
            right_on=["player", "Team_std"],
            how="left",
            suffixes=("", "_receiving"),
        )
        logger.info(f"Merged receiving data: {master_df.shape}")

    # For rushers
    if "rushing_data" in data_dict:
        rush_df = data_dict["rushing_data"].copy()
        master_df = pd.merge(
            master_df,
            rush_df,
            left_on=["Player", "Team_std"],
            right_on=["player", "Team_std"],
            how="left",
            suffixes=("", "_rushing"),
        )
        logger.info(f"Merged rushing data: {master_df.shape}")

    # Add team stats
    if "team_stats" in data_dict:
        team_df = data_dict["team_stats"].copy()
        master_df = pd.merge(
            master_df, team_df, on=["Team_std"], how="left", suffixes=("", "_team")
        )
        logger.info(f"Merged team stats: {master_df.shape}")

    # Calculate half-PPR points if not already present
    if "Half_PPR" not in master_df.columns:
        master_df = calculate_half_ppr_points(master_df)

    # Validate the final dataset
    if not validate_player_performance_data(master_df):
        logger.error("Player performance data validation failed")
        return pd.DataFrame()

    logger.info(f"Master player dataset created with shape: {master_df.shape}")
    return master_df


def calculate_half_ppr_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate half-PPR fantasy points if not already present.

    Args:
        df: Player dataframe

    Returns:
        pd.DataFrame: Dataframe with half-PPR points calculated
    """
    logger.info("Calculating half-PPR fantasy points")

    # Check if points already exist
    if "Half_PPR" in df.columns:
        logger.info("Half-PPR points already exist, skipping calculation")
        return df

    # Calculate points based on available columns
    required_cols = {
        "Passing Yds": ["Passing Yds"],
        "Passing TD": ["Passing TD"],
        "Passing Int": ["Passing Int"],
        "Rushing Yds": ["Rushing Yds"],
        "Rushing TD": ["Rushing TD"],
        "Receiving Rec": ["Receiving Rec"],
        "Receiving Yds": ["Receiving Yds"],
        "Receiving TD": ["Receiving TD"],
        "FL": ["FL"],
    }

    # Check available columns
    missing_cols = []
    for col_set in required_cols.values():
        if not any(col in df.columns for col in col_set):
            missing_cols.extend(col_set)

    if missing_cols:
        logger.warning(f"Missing columns for half-PPR calculation: {missing_cols}")
        return df

    # Fill missing values with 0
    for col_set in required_cols.values():
        for col in col_set:
            if col in df.columns and df[col].isnull().any():
                logger.warning(
                    f"Filling {df[col].isnull().sum()} missing values in {col} with 0"
                )
                df[col] = df[col].fillna(0)

    # Calculate half-PPR points
    df["Half_PPR"] = (
        df["Passing Yds"] / 25
        + df["Passing TD"] * 4
        + df["Passing Int"] * -2
        + df["Rushing Yds"] / 10
        + df["Rushing TD"] * 6
        + df["Receiving Rec"] * 0.5
        + df["Receiving Yds"] / 10
        + df["Receiving TD"] * 6
        + df["FL"] * -2
    )

    # Calculate PPG
    if "G" in df.columns:
        df["Half_PPR_PPG"] = df["Half_PPR"] / df["G"]
        # Replace inf/NaN values
        df["Half_PPR_PPG"] = (
            df["Half_PPR_PPG"].replace([np.inf, -np.inf], np.nan).fillna(0)
        )

    logger.info("Half-PPR points calculation completed")
    return df


def save_processed_data(df: pd.DataFrame, filename: str, output_path: str) -> None:
    """
    Save processed dataframe to CSV file.

    Args:
        df: Dataframe to save
        filename: Output filename
        output_path: Path to save the file
    """
    import os

    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Save dataframe
    full_path = os.path.join(output_path, filename)
    df.to_csv(full_path, index=False)
    logger.info(f"Saved processed data to {full_path}")


def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {str(e)}")
        return pd.DataFrame()
