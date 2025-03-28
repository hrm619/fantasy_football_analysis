import pytest
import pandas as pd
import numpy as np
import os
import sys
import yaml
from unittest.mock import patch, mock_open, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import load_config, load_csv_data, load_all_data, validate_dataframe

# Sample config for testing
SAMPLE_CONFIG = {
    'data_paths': {
        'raw_data': 'data/raw/',
        'processed_data': 'data/processed/'
    },
    'analysis': {
        'season': 2024
    }
}

# Sample CSV data for testing
SAMPLE_CSV_DATA = """Player,Position,Team,ADP
Patrick Mahomes,QB,KC,15.5
Justin Jefferson,WR,MIN,1.8
Christian McCaffrey,RB,SF,1.2
Travis Kelce,TE,KC,12.1
"""

class TestDataLoader:
    
    def test_load_config(self):
        """Test that load_config function works correctly"""
        # Mock the open function to return our sample config
        m = mock_open(read_data=yaml.dump(SAMPLE_CONFIG))
        with patch('builtins.open', m):
            config = load_config('fake_path.yaml')
            
            # Assert the config is loaded correctly
            assert config is not None
            assert 'data_paths' in config
            assert 'raw_data' in config['data_paths']
            assert config['data_paths']['raw_data'] == 'data/raw/'
            assert config['analysis']['season'] == 2024
    
    def test_load_config_file_not_found(self):
        """Test that load_config handles file not found error"""
        # Mock the open function to raise FileNotFoundError
        with patch('builtins.open', side_effect=FileNotFoundError()):
            with pytest.raises(FileNotFoundError):
                load_config('nonexistent_file.yaml')
    
    def test_load_csv_data(self):
        """Test that load_csv_data function works correctly"""
        # Mock pandas.read_csv to return a DataFrame
        df = pd.read_csv(pd.io.common.StringIO(SAMPLE_CSV_DATA))
        
        with patch('pandas.read_csv', return_value=df):
            # Mock the validate_dataframe function
            with patch('src.data.data_loader.validate_dataframe', return_value=True):
                result_df = load_csv_data('fake_path.csv')
                
                # Assert the DataFrame is loaded correctly
                assert result_df is not None
                assert isinstance(result_df, pd.DataFrame)
                assert 'Player' in result_df.columns
                assert len(result_df) == 4
                assert result_df['Position'].tolist() == ['QB', 'WR', 'RB', 'TE']
    
    def test_load_csv_data_file_not_found(self):
        """Test that load_csv_data handles file not found error"""
        # Mock pandas.read_csv to raise FileNotFoundError
        with patch('pandas.read_csv', side_effect=FileNotFoundError()):
            with pytest.raises(Exception):
                load_csv_data('nonexistent_file.csv')
    
    def test_validate_dataframe(self):
        """Test that validate_dataframe function works correctly"""
        # Create a sample DataFrame
        df = pd.DataFrame({
            'Player': ['Player A', 'Player B', 'Player C'],
            'Position': ['QB', 'RB', 'WR'],
            'Team': ['KC', 'SF', 'MIN'],
            'Value': [100, 200, 300]
        })
        
        # Test with valid DataFrame
        assert validate_dataframe(df, 'test_df') == True
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        assert validate_dataframe(empty_df, 'empty_df') == False
        
        # Test with DataFrame containing missing values
        missing_df = df.copy()
        missing_df.loc[0, 'Team'] = None
        # Validation should still pass but with warnings
        assert validate_dataframe(missing_df, 'missing_df') == True
    
    def test_load_all_data(self):
        """Test that load_all_data function works correctly"""
        # Create a sample config
        config = {
            'data_paths': {
                'raw_data': 'data/raw/'
            }
        }
        
        # Create sample DataFrames for different data files
        preseason_df = pd.DataFrame({'Player': ['A', 'B'], 'ADP': [1.0, 2.0]})
        season_df = pd.DataFrame({'Player': ['A', 'B'], 'Points': [100, 200]})
        
        # Mock load_csv_data to return our sample DataFrames
        with patch('src.data.data_loader.load_csv_data', side_effect=[preseason_df, season_df]):
            # Mock os.path.join to return predictable paths
            with patch('os.path.join', side_effect=lambda *args: '/'.join(args)):
                # Only test with a subset of the expected files to keep the test simple
                with patch('src.data.data_loader.files', {'preseason_rankings': 'file1.csv', 'season_data': 'file2.csv'}):
                    result = load_all_data(config)
                    
                    # Assert the result contains the expected keys and DataFrames
                    assert 'preseason_rankings' in result
                    assert 'season_data' in result
                    assert isinstance(result['preseason_rankings'], pd.DataFrame)
                    assert isinstance(result['season_data'], pd.DataFrame)
                    assert len(result['preseason_rankings']) == 2
                    assert len(result['season_data']) == 2
