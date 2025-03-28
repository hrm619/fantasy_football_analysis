import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.performance import calculate_performance_metrics, analyze_expectation_vs_performance
from src.analysis.position_value import calculate_vorp, analyze_positional_scarcity
from src.analysis.draft_value import analyze_adp_vs_performance, calculate_vbd_rankings
from src.analysis.tiering import identify_player_tiers

class TestPerformanceAnalysis:
    
    def setup_method(self):
        """Set up test data for performance analysis"""
        # Create a sample player performance DataFrame
        self.performance_df = pd.DataFrame({
            'Player': ['Player A', 'Player B', 'Player C', 'Player D'],
            'Team_std': ['KC', 'SF', 'MIN', 'DAL'],
            'FantPos': ['QB', 'RB', 'WR', 'TE'],
            'G': [16, 15, 16, 14],
            'Half_PPR': [300, 250, 200, 150],
            'Half_PPR_PPG': [18.75, 16.67, 12.5, 10.71],
            'ADP': [15, 8, 25, 50]
        })
    
    def test_calculate_performance_metrics(self):
        """Test calculation of performance metrics"""
        # Call the function
        result_df = calculate_performance_metrics(self.performance_df)
        
        # Assert basic structure
        assert result_df is not None
        assert len(result_df) == len(self.performance_df)
        
        # Assert positional ranks were calculated
        assert 'QB_Rank' in result_df.columns
        assert 'RB_Rank' in result_df.columns
        assert 'WR_Rank' in result_df.columns
        assert 'TE_Rank' in result_df.columns
        assert 'FLX_Rank' in result_df.columns
        assert 'Overall_Rank' in result_df.columns
        
        # Check rank values
        player_a = result_df[result_df['Player'] == 'Player A'].iloc[0]
        assert player_a['QB_Rank'] == 1
        assert player_a['Overall_Rank'] == 1
        
        player_d = result_df[result_df['Player'] == 'Player D'].iloc[0]
        assert player_d['TE_Rank'] == 1
        assert player_d['FLX_Rank'] == 3  # Should be after RB and WR
    
    def test_analyze_expectation_vs_performance(self):
        """Test analysis of expectations vs performance"""
        # Add positional ranks for testing
        test_df = self.performance_df.copy()
        test_df['QB_Rank'] = [1, np.nan, np.nan, np.nan]
        test_df['RB_Rank'] = [np.nan, 1, np.nan, np.nan]
        test_df['WR_Rank'] = [np.nan, np.nan, 1, np.nan]
        test_df['TE_Rank'] = [np.nan, np.nan, np.nan, 1]
        test_df['Overall_Rank'] = [1, 2, 3, 4]
        
        # Add preseason rank expectations
        test_df['ADP Pos Rank'] = ['QB2', 'RB1', 'WR3', 'TE5']
        
        # Call the function
        result_df = analyze_expectation_vs_performance(test_df)
        
        # Assert results
        assert result_df is not None
        assert 'ADP_vs_Actual_Rank_Delta' in result_df.columns
        
        # Player B has ADP of 8 but is ranked 2 overall (overperformed)
        player_b = result_df[result_df['Player'] == 'Player B'].iloc[0]
        assert player_b['ADP_vs_Actual_Rank_Delta'] > 0
        
        # Player C has ADP of 25 but is ranked 3 overall (overperformed slightly)
        player_c = result_df[result_df['Player'] == 'Player C'].iloc[0]
        assert player_c['ADP_vs_Actual_Rank_Delta'] > 0
        
        # Performance categorization
        assert 'Performance_Category' in result_df.columns


class TestPositionValueAnalysis:
    
    def setup_method(self):
        """Set up test data for position value analysis"""
        # Create sample player data with multiple players per position
        players = []
        # QBs
        for i in range(20):
            players.append({
                'Player': f'QB{i+1}',
                'Team_std': f'Team{i%10}',
                'FantPos': 'QB',
                'Half_PPR': 300 - i*10,
                'G': 16
            })
        # RBs
        for i in range(40):
            players.append({
                'Player': f'RB{i+1}',
                'Team_std': f'Team{i%10}',
                'FantPos': 'RB',
                'Half_PPR': 280 - i*5,
                'G': 16 - i//20
            })
        # WRs
        for i in range(40):
            players.append({
                'Player': f'WR{i+1}',
                'Team_std': f'Team{i%10}',
                'FantPos': 'WR',
                'Half_PPR': 260 - i*4,
                'G': 16 - i//20
            })
        # TEs
        for i in range(20):
            players.append({
                'Player': f'TE{i+1}',
                'Team_std': f'Team{i%10}',
                'FantPos': 'TE',
                'Half_PPR': 200 - i*8,
                'G': 16 - i//10
            })
        
        self.player_df = pd.DataFrame(players)
        
        # Define baseline values
        self.baselines = {
            'QB': 12,
            'RB': 24,
            'WR': 24,
            'TE': 12
        }
    
    def test_calculate_vorp(self):
        """Test Value Over Replacement Player calculation"""
        # Call the function
        result_df = calculate_vorp(self.player_df, self.baselines)
        
        # Assert VORP columns exist
        assert 'VORP' in result_df.columns
        assert 'VORP_Per_Game' in result_df.columns
        assert 'VORP_Rank' in result_df.columns
        
        # Check VORP values
        # QB1 should have highest VORP
        qb1 = result_df[result_df['Player'] == 'QB1'].iloc[0]
        assert qb1['VORP'] > 0
        
        # QB at baseline should have VORP of 0
        qb_baseline = result_df[result_df['Player'] == f'QB{self.baselines["QB"]}'].iloc[0]
        assert qb_baseline['VORP'] == 0
        
        # RB1 should have positive VORP
        rb1 = result_df[result_df['Player'] == 'RB1'].iloc[0]
        assert rb1['VORP'] > 0
        
        # Verify VORP_Per_Game calculation
        assert abs(qb1['VORP_Per_Game'] - (qb1['VORP'] / qb1['G'])) < 0.001
        
        # Check rank ordering
        assert result_df.loc[result_df['VORP_Rank'] == 1, 'VORP'].values[0] == result_df['VORP'].max()
    
    def test_analyze_positional_scarcity(self):
        """Test positional scarcity analysis"""
        # First calculate VORP for proper ranking
        player_df_with_vorp = calculate_vorp(self.player_df, self.baselines)
        
        # Call the function
        scarcity_results = analyze_positional_scarcity(player_df_with_vorp)
        
        # Assert results structure
        assert isinstance(scarcity_results, dict)
        
        # Check for tier results
        for pos in ['QB', 'RB', 'WR', 'TE']:
            tier_key = f"{pos}_tiers"
            diff_key = f"{pos}_rank_diffs"
            
            assert tier_key in scarcity_results or pos not in self.player_df['FantPos'].unique()
            assert diff_key in scarcity_results or pos not in self.player_df['FantPos'].unique()
            
            if tier_key in scarcity_results:
                tier_df = scarcity_results[tier_key]
                assert 'Position' in tier_df.columns
                assert 'Tier' in tier_df.columns
                assert 'Avg_Points' in tier_df.columns
        
        # Check position value adjustments
        assert 'position_value_adjustments' in scarcity_results
        adj_df = scarcity_results['position_value_adjustments']
        assert 'Position' in adj_df.columns
        assert 'Value_Adjustment_Factor' in adj_df.columns
        
        # Sum of adjustment factors should be close to number of positions
        assert abs(adj_df['Value_Adjustment_Factor'].sum() - len(self.baselines)) < 0.01


class TestDraftValueAnalysis:
    
    def setup_method(self):
        """Set up test data for draft value analysis"""
        # Create sample player data with ADP values
        players = []
        for round_num in range(1, 13):  # 12 rounds
            for pick in range(1, 13):  # 12 picks per round
                adp = (round_num - 1) * 12 + pick
                # Performance tends to decrease by round but with variance
                perf_base = max(300 - (round_num - 1) * 20, 50)
                perf_var = np.random.normal(0, 20)
                half_ppr = max(perf_base + perf_var, 10)  # Ensure minimum of 10 points
                
                # Distribute positions across rounds
                if round_num == 1:
                    pos = 'RB' if pick <= 6 else 'WR'
                elif round_num == 2:
                    pos = 'WR' if pick <= 6 else 'RB'
                elif round_num == 3:
                    pos = 'TE' if pick <= 3 else ('QB' if pick <= 6 else ('RB' if pick <= 9 else 'WR'))
                else:
                    pos = np.random.choice(['QB', 'RB', 'WR', 'TE'], p=[0.15, 0.35, 0.35, 0.15])
                
                players.append({
                    'Player': f'Player{adp}',
                    'FantPos': pos,
                    'ADP': adp,
                    'Half_PPR': half_ppr,
                    'Half_PPR_PPG': half_ppr / 16
                })
        
        self.player_df = pd.DataFrame(players)
    
    def test_analyze_adp_vs_performance(self):
        """Test ADP vs performance analysis"""
        # Call the function
        adp_results = analyze_adp_vs_performance(self.player_df, 12)
        
        # Assert results structure
        assert isinstance(adp_results, dict)
        assert 'round_stats' in adp_results
        assert 'position_round_stats' in adp_results
        assert 'optimal_positions_by_round' in adp_results
        
        # Check round stats
        round_stats = adp_results['round_stats']
        assert 'Draft_Round' in round_stats.columns
        assert 'Avg_Points' in round_stats.columns
        assert 'Success_Rate' in round_stats.columns
        assert 'Bust_Rate' in round_stats.columns
        
        # Verify round ordering - early rounds should have higher average points
        assert round_stats.loc[round_stats['Draft_Round'] == 1, 'Avg_Points'].values[0] > \
               round_stats.loc[round_stats['Draft_Round'] == 12, 'Avg_Points'].values[0]
        
        # Check position round stats
        pos_round_stats = adp_results['position_round_stats']
        assert 'Position' in pos_round_stats.columns
        assert 'Draft_Round' in pos_round_stats.columns
        assert 'Avg_Points' in pos_round_stats.columns
        
        # Check optimal positions
        opt_positions = adp_results['optimal_positions_by_round']
        assert 'Draft_Round' in opt_positions.columns
        assert 'Optimal_Position' in opt_positions.columns
        
        # At least one round should identify optimal position
        assert len(opt_positions) > 0
    
    def test_calculate_vbd_rankings(self):
        """Test Value Based Drafting ranking calculation"""
        # Set up sample VORP data
        player_df = self.player_df.copy()
        player_df['VORP'] = player_df['Half_PPR'] - player_df.groupby('FantPos')['Half_PPR'].transform(
            lambda x: x.sort_values(ascending=False).iloc[11] if len(x) > 11 else 0
        )
        player_df['VORP'] = player_df['VORP'].clip(lower=0)
        
        # Calculate VBD rankings
        vbd_df = calculate_vbd_rankings(player_df)
        
        # Assert VBD columns exist
        assert 'VBD' in vbd_df.columns
        assert 'VBD_Rank' in vbd_df.columns
        
        # Top VBD player should have highest value
        top_vbd = vbd_df.sort_values('VBD', ascending=False).iloc[0]
        assert top_vbd['VBD_Rank'] == 1
        
        # If ADP exists, VBD per ADP should exist
        if 'ADP' in vbd_df.columns:
            assert 'VBD_per_ADP' in vbd_df.columns
            
            # Higher VBD relative to ADP indicates better value
            top_value = vbd_df.sort_values('VBD_per_ADP', ascending=False).iloc[0]
            assert top_value['VBD_per_ADP'] > 0
        
        # Draft round should be calculated if ADP exists
        if 'ADP' in vbd_df.columns:
            assert 'Draft_Round' in vbd_df.columns
            assert max(vbd_df['Draft_Round']) <= 12  # 12-team league


class TestTieringAnalysis:
    
    def setup_method(self):
        """Set up test data for tiering analysis"""
        # Create sample player data with clear tiers
        self.player_df = pd.DataFrame({
            'Player': [f'Player{i+1}' for i in range(50)],
            'Team_std': [f'Team{i%10}' for i in range(50)],
            'FantPos': ['QB'] * 10 + ['RB'] * 15 + ['WR'] * 15 + ['TE'] * 10,
            'Half_PPR': 
                # QBs with 3 tiers
                [350, 340, 330, 290, 280, 275, 220, 210, 200, 195] + 
                # RBs with 3 tiers
                [320, 310, 300, 290, 280, 240, 230, 220, 210, 200, 160, 150, 140, 130, 120] +
                # WRs with 3 tiers
                [300, 290, 280, 270, 260, 220, 210, 200, 190, 180, 140, 130, 120, 110, 100] +
                # TEs with 2 tiers
                [250, 240, 230, 160, 150, 140, 130, 120, 110, 100],
            'Half_PPR_PPG': 
                # QBs PPG
                [21.9, 21.3, 20.6, 18.1, 17.5, 17.2, 13.8, 13.1, 12.5, 12.2] +
                # RBs PPG
                [20.0, 19.4, 18.8, 18.1, 17.5, 15.0, 14.4, 13.8, 13.1, 12.5, 10.0, 9.4, 8.8, 8.1, 7.5] +
                # WRs PPG
                [18.8, 18.1, 17.5, 16.9, 16.3, 13.8, 13.1, 12.5, 11.9, 11.3, 8.8, 8.1, 7.5, 6.9, 6.3] +
                # TEs PPG
                [15.6, 15.0, 14.4, 10.0, 9.4, 8.8, 8.1, 7.5, 6.9, 6.3]
        })
    
    def test_identify_player_tiers(self):
        """Test player tier identification"""
        # Call the function
        tier_results = identify_player_tiers(self.player_df, min_clusters=2, max_clusters=5)
        
        # Assert results structure
        assert isinstance(tier_results, dict)
        
        # Check for tier results for each position
        for pos in ['QB', 'RB', 'WR', 'TE']:
            tier_key = f"{pos}_tiers"
            stats_key = f"{pos}_tier_stats"
            
            assert tier_key in tier_results
            assert stats_key in tier_results
            
            # Check tier dataframe
            tier_df = tier_results[tier_key]
            assert 'Tier' in tier_df.columns
            assert 'Tier_Cluster' in tier_df.columns
            
            # Check stats dataframe
            stats_df = tier_results[stats_key]
            assert 'Tier' in stats_df.columns
            assert 'Avg_Points' in stats_df.columns
            assert 'Player_Count' in stats_df.columns
            
            # Verify proper tiering - Tier 1 should have highest average points
            tier_1 = stats_df[stats_df['Tier'] == 'Tier 1']
            tier_2 = stats_df[stats_df['Tier'] == 'Tier 2'] if 'Tier 2' in stats_df['Tier'].values else None
            
            if not tier_1.empty and tier_2 is not None and not tier_2.empty:
                assert tier_1['Avg_Points'].values[0] > tier_2['Avg_Points'].values[0]
