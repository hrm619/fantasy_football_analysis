import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame, name: str) -> bool:
    """
    Validate a dataframe to ensure it meets expected criteria.

    Args:
        df: DataFrame to validate
        name: Name of the dataframe for logging

    Returns:
        bool: True if validation passes, False otherwise
    """
    if df is None:
        logger.error(f"DataFrame '{name}' is None")
        return False

    if df.empty:
        logger.warning(f"DataFrame '{name}' is empty")
        return False

    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        logger.warning(f"Missing values detected in '{name}':")
        for col, count in missing_counts[missing_counts > 0].items():
            pct_missing = (count / len(df)) * 100
            if pct_missing > 10:  # Only warn for significant missing data
                logger.warning(
                    f"  - {col}: {count} missing values ({pct_missing:.1f}%)"
                )

    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate rows in '{name}'")

    # Log basic statistics
    logger.info(
        f"DataFrame '{name}' validated: {df.shape[0]} rows, {df.shape[1]} columns"
    )
    return True


def validate_analysis_output(result: Any, name: str) -> bool:
    """
    Validate analysis output of various types.

    Args:
        result: Analysis result to validate
        name: Name of the analysis for logging

    Returns:
        bool: True if validation passes, False otherwise
    """
    if result is None:
        logger.error(f"Analysis result '{name}' is None")
        return False

    # Validate dataframe
    if isinstance(result, pd.DataFrame):
        return validate_dataframe(result, name)

    # Validate dictionary of dataframes
    if isinstance(result, dict):
        if not result:
            logger.warning(f"Dictionary result '{name}' is empty")
            return False

        all_valid = True
        for key, value in result.items():
            if isinstance(value, pd.DataFrame):
                if not validate_dataframe(value, f"{name}.{key}"):
                    all_valid = False

        return all_valid

    # Other types
    logger.info(f"Analysis result '{name}' of type {type(result)} validated")
    return True


def validate_numeric_columns(df: pd.DataFrame, columns: List[str], name: str) -> bool:
    """
    Validate numeric columns in a dataframe.

    Args:
        df: DataFrame to validate
        columns: List of column names expected to be numeric
        name: Name of the dataframe for logging

    Returns:
        bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        logger.error(f"DataFrame '{name}' is None or empty")
        return False

    # Check each column
    all_valid = True
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in '{name}'")
            all_valid = False
            continue

        # Check numeric type
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(
                f"Column '{col}' in '{name}' is not numeric, type: {df[col].dtype}"
            )
            all_valid = False
            continue

        # Check for infinite values
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            logger.warning(
                f"Column '{col}' in '{name}' has {inf_count} infinite values"
            )
            all_valid = False

        # Log basic statistics
        logger.info(
            f"Column '{col}' in '{name}': min={df[col].min()}, max={df[col].max()}, mean={df[col].mean()}"
        )

    return all_valid


def validate_player_performance_data(df: pd.DataFrame) -> bool:
    """
    Validate player performance data specifically for VORP analysis.
    
    Args:
        df: DataFrame containing player performance data
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        logger.error("Player performance data is None or empty")
        return False
        
    # Core columns required for VORP analysis
    required_columns = {
        'Player': str,
        'FantPos': str,
        'Team': str,
        'Half_PPR': float,
        'G': int
    }
    
    # Check required columns exist and have correct types
    all_valid = True
    for col, expected_type in required_columns.items():
        if col not in df.columns:
            logger.error(f"Required column '{col}' missing from player performance data")
            all_valid = False
            continue
            
        if not pd.api.types.is_dtype_equal(df[col].dtype, expected_type):
            logger.warning(f"Column '{col}' has incorrect type: {df[col].dtype}, expected {expected_type}")
            all_valid = False
            
    # Validate position values
    valid_positions = {'QB', 'RB', 'WR', 'TE'}
    invalid_positions = set(df['FantPos'].unique()) - valid_positions
    if invalid_positions:
        logger.warning(f"Invalid positions found: {invalid_positions}")
        all_valid = False
        
    # Validate games played
    invalid_games = df[~df['G'].between(0, 17)]  # NFL season is 17 games
    if not invalid_games.empty:
        logger.warning(f"Found {len(invalid_games)} players with invalid games played values")
        logger.warning(f"Invalid games range: {invalid_games['G'].min()} to {invalid_games['G'].max()}")
        all_valid = False
        
    # Validate Half_PPR points
    if df['Half_PPR'].isnull().any():
        logger.error(f"Found {df['Half_PPR'].isnull().sum()} missing Half_PPR values")
        all_valid = False
        
    # Check for negative points
    negative_points = df[df['Half_PPR'] < 0]
    if not negative_points.empty:
        logger.warning(f"Found {len(negative_points)} players with negative Half_PPR points")
        all_valid = False
        
    # Log summary statistics
    logger.info("\nPlayer Performance Data Summary:")
    logger.info(f"Total players: {len(df)}")
    logger.info("\nPlayers by position:")
    for pos in valid_positions:
        count = len(df[df['FantPos'] == pos])
        logger.info(f"  {pos}: {count} players")
        
    logger.info("\nPoints distribution:")
    logger.info(f"  Min: {df['Half_PPR'].min():.2f}")
    logger.info(f"  Max: {df['Half_PPR'].max():.2f}")
    logger.info(f"  Mean: {df['Half_PPR'].mean():.2f}")
    logger.info(f"  Median: {df['Half_PPR'].median():.2f}")
    
    return all_valid
