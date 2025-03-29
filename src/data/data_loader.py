import pandas as pd
import os
import yaml
import logging
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def load_csv_data(file_path: str, validate: bool = True) -> pd.DataFrame:
    """
    Load CSV data into a pandas DataFrame with validation.

    Args:
        file_path: Path to the CSV file
        validate: Whether to validate the data after loading

    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)

        # Clean team data if it's the team stats file
        if "Team Stats" in file_path:
            # Remove empty rows
            df = df.dropna(subset=['Team (Full)', 'Team (Alt)', 'Team'])
            df = df[df['Team (Full)'].str.strip() != '']
            
            # Convert numeric columns
            numeric_cols = ['PF', 'Yds', 'Passing Att', 'Passing Yds', 'Rushing Att', 'Rushing Yds']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"Cleaned team data shape: {df.shape}")

        # Log data shape
        logger.info(f"Loaded dataframe with shape: {df.shape}")

        if validate:
            validate_dataframe(df, file_path)

        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def validate_dataframe(df: pd.DataFrame, file_path: str) -> None:
    """
    Validate dataframe to ensure it meets expected criteria.

    Args:
        df: DataFrame to validate
        file_path: Original file path for error messages
    """
    # Check for empty dataframe
    if df.empty:
        logger.warning(f"DataFrame from {file_path} is empty")

    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        logger.warning(f"Missing values detected in {file_path}:")
        for col, count in missing_counts[missing_counts > 0].items():
            logger.warning(f"  - {col}: {count} missing values")

    # Check data types for key columns based on filename
    filename = os.path.basename(file_path)

    if "PreSeason Rankings" in filename:
        expected_columns = ["Player", "Position", "Team", "ADP"]
        for col in expected_columns:
            if col not in df.columns:
                logger.warning(f"Expected column {col} not found in {filename}")

    elif "Season Player Data" in filename:
        expected_columns = ["Player", "Team", "FantPos", "G"]
        for col in expected_columns:
            if col not in df.columns:
                logger.warning(f"Expected column {col} not found in {filename}")

    # Log successful validation
    logger.info(f"Validation completed for {file_path}")


def load_all_data(config: dict) -> Dict[str, pd.DataFrame]:
    """
    Load all required datasets based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of loaded dataframes
    """
    raw_data_path = config["data_paths"]["raw_data"]

    # Define files to load
    files = {
        "preseason_rankings": "F00  2024 PreSeason Rankings.csv",
        "season_data": "F01  2023 2024 Season Player Data.csv",
        "passing_data": "F02  2023 2024 pff_passing_summary.csv",
        "line_data": "F03  2023 2024 pff_line_pass_blocking_efficiency.csv",
        "receiving_data": "F04  2023 2024 pff_receiving_summary.csv",
        "rushing_data": "F05  2023 2024 pff_rushing_summary.csv",
        "team_stats": "F06  2024 2023 Team Stats.csv"
    }

    # Load each file
    data_dict = {}
    for key, filename in files.items():
        file_path = os.path.join(raw_data_path, filename)
        data_dict[key] = load_csv_data(file_path)

    logger.info(f"Successfully loaded {len(files)} datasets")
    return data_dict
