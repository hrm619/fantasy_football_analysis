import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def standardize_team_names_across_datasets(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Standardize team names across all datasets.
    
    Args:
        data_dict: Dictionary of dataframes to process
        
    Returns:
        Dict[str, pd.DataFrame]: Processed dataframes with standardized team names
    """
    # Check team name variations
    logger.info("Team name variations across datasets:")
    team_variations = {}
    
    for key, df in data_dict.items():
        if 'Team' in df.columns:
            teams = df['Team'].dropna().unique()  # Drop NaN values
            team_variations[key] = teams
        elif 'team_name' in df.columns:
            teams = df['team_name'].dropna().unique()  # Drop NaN values
            team_variations[key] = teams
    
    for dataset, teams in team_variations.items():
        logger.info(f"\n{dataset}: {sorted(teams.tolist())}")
    
    # Create a mapping dictionary from the preseason rankings
    if "preseason_rankings" in data_dict:
        team_mapping = {}
        df = data_dict["preseason_rankings"]
        if all(col in df.columns for col in ["Team", "Team (Alt)", "Team (Full)"]):
            for idx, row in (
                df[["Team", "Team (Alt)", "Team (Full)"]].drop_duplicates().iterrows()
            ):
                team_mapping[row["Team"]] = row["Team"]
                team_mapping[row["Team (Alt)"]] = row["Team"]
                team_mapping[row["Team (Full)"]] = row["Team"]

        # Apply mapping to all dataframes
        for key, df in data_dict.items():
            if "Team" in df.columns:
                data_dict[key]["Team_std"] = (
                    df["Team"].map(team_mapping).fillna(df["Team"])
                )
            elif "team_name" in df.columns:
                data_dict[key]["Team_std"] = (
                    df["team_name"].map(team_mapping).fillna(df["team_name"])
                )

            # Validate mapping
            if "Team_std" in data_dict[key].columns:
                missing_mappings = data_dict[key][data_dict[key]["Team_std"].isnull()]
                if not missing_mappings.empty:
                    logger.warning(
                        f"Missing team mappings in {key} dataset: {missing_mappings['Team'].unique() if 'Team' in missing_mappings.columns else missing_mappings['team_name'].unique()}"
                    )
    
    # Verify standardization
    logger.info("Verifying team name standardization:")
    for key, df in data_dict.items():
        if 'Team_std' in df.columns:
            std_teams = df['Team_std'].dropna().unique()  # Drop NaN values
            logger.info(f"\n{key}: {sorted(std_teams.tolist())}")
    
    return data_dict 