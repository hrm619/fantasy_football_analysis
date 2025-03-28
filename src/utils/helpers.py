import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Union, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_z_scores(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Calculate z-scores for specified columns.
    
    Args:
        df: Input dataframe
        columns: List of columns to calculate z-scores for
        
    Returns:
        pd.DataFrame: Dataframe with added z-score columns
    """
    result_df = df.copy()
    
    for col in columns:
        if col in df.columns:
            z_col = f"{col}_Z"
            result_df[z_col] = (df[col] - df[col].mean()) / df[col].std()
    
    return result_df

def calculate_percentiles(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Calculate percentiles for specified columns.
    
    Args:
        df: Input dataframe
        columns: List of columns to calculate percentiles for
        
    Returns:
        pd.DataFrame: Dataframe with added percentile columns
    """
    result_df = df.copy()
    
    for col in columns:
        if col in df.columns:
            pct_col = f"{col}_Percentile"
            result_df[pct_col] = df[col].rank(pct=True) * 100
    
    return result_df

def detect_outliers(df: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in a column using IQR method.
    
    Args:
        df: Input dataframe
        column: Column to check for outliers
        threshold: IQR multiplier for outlier detection
        
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in dataframe")
        return pd.Series(False, index=df.index)
    
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - (threshold * iqr)
    upper_bound = q3 + (threshold * iqr)
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def format_player_name(name: str) -> str:
    """
    Format player name consistently.
    
    Args:
        name: Raw player name
        
    Returns:
        str: Formatted player name
    """
    if not isinstance(name, str):
        return str(name)
    
    # Remove special characters
    formatted = ''.join(c for c in name if c.isalnum() or c.isspace() or c == '.')
    
    # Handle common suffixes
    suffixes = ['Jr.', 'Sr.', 'II', 'III', 'IV']
    for suffix in suffixes:
        formatted = formatted.replace(f" {suffix}", f".{suffix}")
    
    return formatted.strip()

def standardize_team_abbr(abbr: str, team_mapping: Optional[Dict[str, str]] = None) -> str:
    """
    Standardize team abbreviations.
    
    Args:
        abbr: Team abbreviation
        team_mapping: Optional mapping of team abbreviations
        
    Returns:
        str: Standardized team abbreviation
    """
    if not isinstance(abbr, str):
        return str(abbr)
    
    # Default mappings
    default_mapping = {
        'JAC': 'JAX', 'JAG': 'JAX',
        'KCC': 'KC', 'KAN': 'KC',
        'LA': 'LAR', 'RAM': 'LAR',
        'LAC': 'LAC', 'SDG': 'LAC',
        'LVR': 'LV', 'LAS': 'LV', 'OAK': 'LV',
        'NE': 'NE', 'NWE': 'NE',
        'NO': 'NO', 'NOR': 'NO',
        'SF': 'SF', 'SFO': 'SF',
        'TB': 'TB', 'TAM': 'TB',
        'WSH': 'WAS', 'WFT': 'WAS'
    }
    
    # Use provided mapping if available
    mapping = team_mapping if team_mapping is not None else default_mapping
    
    # Apply mapping
    return mapping.get(abbr.upper(), abbr.upper())

def categorize_adp_vs_performance(df: pd.DataFrame, adp_col: str, rank_col: str, bins: List[float] = None) -> pd.DataFrame:
    """
    Categorize players based on ADP vs. actual performance.
    
    Args:
        df: Player dataframe
        adp_col: ADP column name
        rank_col: Actual rank column name
        bins: Optional bins for categorization
        
    Returns:
        pd.DataFrame: Dataframe with added performance category
    """
    result_df = df.copy()
    
    # Create ADP rank if needed
    if adp_col in result_df.columns and rank_col in result_df.columns:
        # Calculate delta
        result_df['ADP_Performance_Delta'] = result_df[adp_col] - result_df[rank_col]
        
        # Use default bins if none provided
        if bins is None:
            bins = [-float('inf'), -12, -6, 6, 12, float('inf')]
            labels = ['Significant Overperformer', 'Moderate Overperformer', 
                     'Met Expectations', 'Moderate Underperformer', 'Significant Underperformer']
        else:
            # Generate labels based on bins
            labels = []
            for i in range(len(bins) - 1):
                if bins[i] < 0 and bins[i+1] < 0:
                    labels.append(f"Overperformer ({bins[i]}-{bins[i+1]})")
                elif bins[i] > 0 and bins[i+1] > 0:
                    labels.append(f"Underperformer ({bins[i]}-{bins[i+1]})")
                else:
                    labels.append("Met Expectations")
        
        # Categorize
        result_df['Performance_Category'] = pd.cut(
            result_df['ADP_Performance_Delta'],
            bins=bins,
            labels=labels
        )
    
    return result_df