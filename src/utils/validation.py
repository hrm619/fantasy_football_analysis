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
