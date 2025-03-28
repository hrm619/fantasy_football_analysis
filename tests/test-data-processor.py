import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, mock_open, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_processor import (
    standardize_team_names,
    filter_season_data,
    create_master_player_dataset,
    calculate_half_ppr_points,
    save_processed_data
)

class TestDataProcessor:
    
    def test_standardize_team_names(self):
        """Test standardizing team names across dataframes"""
        # Create sample DataFrames with team name variations
        preseason_df = pd.DataFrame({
            'Player': ['Player A', 'Player B', 'Player C'],
            'Team': ['KC', 'SF', 'MIN'],
            'Team (Alt)': ['KCC', 'SFO', 'MIN'],
            'Team (Full)': ['Kansas City Chiefs', 'San Francisco 49ers', 'Minnesota Vikings']
        })
        
        season_df = pd.DataFrame({
            'Player': ['Player A', 'Player B', 'Player D'],
            'Team': ['KCC', 'SF', 'DAL']
        })
        
        passing_df = pd.DataFrame({
            'player': ['Player A', 'Player E'],
            'team_name': ['Kansas City', 'Green Bay Packers']
        })
        
        # Create data dictionary
        data_dict = {
            'preseason_rankings': preseason_df,
            'season_data': season_df,
            'passing_data': passing_df
        }
        
        # Standardize team names
        result_dict = standardize_team_names(data_dict)
        
        # Assert all dataframes have the Team_std column
        assert 'Team_std' in result_dict['preseason_rankings'].columns
        assert 'Team_std' in result_dict['season_data'].columns
        assert 'Team_std' in result_dict['passing_data'].columns
        
        # Assert team names are standardized
        assert result_dict['preseason_rankings'].loc[0, 'Team_std'] == 'KC'
        assert result_dict['season_data'].loc[0, 'Team_std'] == 'KC'
        
        # For less common variations, it might keep the original
        # But the important thing is consistency within the dataset
        unique_teams = set()
        for df in result_dict.values():
            unique_teams.update(df['Team_std'].unique())
        
        # Assert we have a reasonable number of unique teams
        assert 10 <= len(unique_teams) <= 32  # NFL has 32 teams
    
    def test_filter_season_data(self):
        """Test filtering data for target season"""
        # Create sample DataFrames with multiple seasons
        preseason_df = pd.DataFrame({
            'Player': ['Player A', 'Player B', 'Player C'],
            'Season': [2024, 2024, 2024],
            'ADP': [1.0, 2.0, 3.0]
        })
        
        season_df = pd.DataFrame({
            'Player': ['Player A', 'Player B', 'Player A', 'Player B'],
            'Season': [2023, 2023, 2024, 2024],
            'Points': [90, 180, 100, 200]
        })
        
        passing_df = pd.DataFrame({
            'player': ['Player A', 'Player B', 'Player A', 'Player B'],
            'season': [2023, 2023, 2024, 2024],
            'yards': [3000, 3500, 3200, 3700]
        })
        
        # Create data dictionary
        data_dict = {
            'preseason_rankings': preseason_df,
            'season_data': season_df,
            'passing_data': passing_df
        }
        
        # Filter for 2024 season
        result_dict = filter_season_data(data_dict, 2024)
        
        # Assert only 2024 data remains
        assert len(result_dict['preseason_rankings']) == 3  # Already only 2024
        assert len(result_dict['season_data']) == 2  # Filtered from 4 to 2
        assert len(result_dict['passing_data']) == 2  # Filtered from 4 to 2
        
        # Verify the season values
        assert all(result_dict['season_data']['Season'] == 2024)
        assert all(result_dict['passing_data']['season'] == 2024)
    
    def test_create_master_player_dataset(self):
        """Test creating master player dataset"""
        # Create sample DataFrames
        season_df = pd.DataFrame({
            'Player': ['Player A', 'Player B', 'Player C'],
            'Team_std': ['KC', 'SF', 'MIN'],
            'FantPos': ['QB', 'RB', 'WR'],
            'G': [16, 15, 14]
        })
        
        preseason_df = pd.DataFrame({
            'Player': ['Player A', 'Player B', 'Player D'],
            'Team_std': ['KC', 'SF', 'DAL'],
            'ADP': [15.0, 8.0, 30.0]
        })
        
        passing_df = pd.DataFrame({
            'player': ['Player A', 'Player E'],
            'Team_std': ['KC', 'GB'],
            'attempts': [500, 450],
            'yards': [4000, 3500]
        })
        
        # Create data dictionary
        data_dict = {
            'season_data': season_df,
            'preseason_rankings': preseason_df,
            'passing_data': passing_df
        }
        
        # Create master dataset
        master_df = create_master_player_dataset(data_dict)
        
        # Assert the master dataset was created correctly
        assert master_df is not None
        assert len(master_df) == len(season_df)  # Should match the base dataset
        
        # Assert it contains columns from both datasets
        assert 'Player' in master_df.columns
        assert 'Team_std' in master_df.columns
        assert 'FantPos' in master_df.columns
        assert 'G' in master_df.columns
        
        # Check that merge worked for Player A
        player_a = master_df[master_df['Player'] == 'Player A'].iloc[0]
        assert player_a['ADP'] == 15.0
        assert player_a['attempts'] == 500
        assert player_a['yards'] == 4000
        
        # Player C should have NaN for ADP (not in preseason_df)
        player_c = master_df[master_df['Player'] == 'Player C'].iloc[0]
        assert pd.isna(player_c['ADP'])
    
    def test_calculate_half_ppr_points(self):
        """Test calculating half-PPR fantasy points"""
        # Create sample DataFrame with necessary stats
        df = pd.DataFrame({
            'Player': ['Player A', 'Player B', 'Player C'],
            'FantPos': ['QB', 'RB', 'WR'],
            'G': [16, 15, 14],
            'Passing Yds': [4000, 0, 0],
            'Passing TD': [30, 0, 0],
            'Passing Int': [10, 0, 0],
            'Rushing Yds': [200, 1200, 50],
            'Rushing TD': [2, 10, 0],
            'Receiving Rec': [0, 40, 80],
            'Receiving Yds': [0, 300, 1200],
            'Receiving TD': [0, 2, 8],
            'FL': [2, 3, 1]
        })
        
        # Calculate half-PPR points
        result_df = calculate_half_ppr_points(df)
        
        # Assert the calculation was done correctly
        assert 'Half_PPR' in result_df.columns
        assert 'Half_PPR_PPG' in result_df.columns
        
        # QB calculation: (4000/25) + (30*4) - (10*2) + (200/10) + (2*6) - (2*2) = 160 + 120 - 20 + 20 + 12 - 4 = 288
        assert abs(result_df.loc[0, 'Half_PPR'] - 288) < 0.1
        
        # RB calculation: (1200/10) + (10*6) + (40*0.5) + (300/10) + (2*6) - (3*2) = 120 + 60 + 20 + 30 + 12 - 6 = 236
        assert abs(result_df.loc[1, 'Half_PPR'] - 236) < 0.1
        
        # WR calculation: (50/10) + (80*0.5) + (1200/10) + (8*6) - (1*2) = 5 + 40 + 120 + 48 - 2 = 211
        assert abs(result_df.loc[2, 'Half_PPR'] - 211) < 0.1
        
        # PPG calculation
        assert abs(result_df.loc[0, 'Half_PPR_PPG'] - (288/16)) < 0.1
        assert abs(result_df.loc[1, 'Half_PPR_PPG'] - (236/15)) < 0.1
        assert abs(result_df.loc[2, 'Half_PPR_PPG'] - (211/14)) < 0.1
    
    def test_save_processed_data(self):
        """Test saving processed data"""
        # Create a sample DataFrame
        df = pd.DataFrame({
            'Player': ['Player A', 'Player B'],
            'Points': [100, 200]
        })
        
        # Mock os.makedirs to prevent actual directory creation
        with patch('os.makedirs'):
            # Mock pandas.DataFrame.to_csv to prevent actual file writing
            with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
                save_processed_data(df, 'test_file.csv', 'test_path')
                
                # Assert to_csv was called with the correct arguments
                mock_to_csv.assert_called_once()
                args, kwargs = mock_to_csv.call_args
                assert 'test_path/test_file.csv' in args[0]
                assert kwargs.get('index') == False
