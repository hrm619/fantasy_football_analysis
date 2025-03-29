import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from scipy.stats import pearsonr

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_team_performance_correlation(player_df: pd.DataFrame, team_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze correlations between team performance metrics and player fantasy production.
    
    Args:
        player_df: Player performance dataframe
        team_df: Team stats dataframe
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of dataframes with team correlation analysis
    """
    logger.info("Analyzing team performance correlations")
    
    # Clean team data
    team_df = team_df.dropna(subset=['Team (Full)', 'Team (Alt)', 'Team'])  # Remove rows with missing team names
    team_df = team_df[team_df['Team (Full)'].str.strip() != '']  # Remove empty team names
    
    # Log input data shapes
    logger.info(f"Input player_df shape: {player_df.shape}")
    logger.info(f"Input team_df shape: {team_df.shape}")
    
    # Log available columns
    logger.info(f"Player DataFrame columns: {player_df.columns.tolist()}")
    logger.info(f"Team DataFrame columns: {team_df.columns.tolist()}")
    
    # Ensure team dataframes has standardized team names
    if 'Team_std' not in team_df.columns:
        # Try different team name columns
        team_name_cols = ['Team', 'Team (Alt)', 'Team (Full)', 'team_name']
        for col in team_name_cols:
            if col in team_df.columns:
                team_df['Team_std'] = team_df[col]
                logger.info(f"Using {col} for team name standardization")
                break
        else:
            logger.error("No team name column found in team_df")
            return {}
    
    # Ensure player_df has standardized team names
    if 'Team_std' not in player_df.columns:
        # Try different team name columns
        team_name_cols = ['Team', 'Team (Alt)', 'Team (Full)', 'team_name']
        for col in team_name_cols:
            if col in player_df.columns:
                player_df['Team_std'] = player_df[col]
                logger.info(f"Using {col} for player team name standardization")
                break
        else:
            logger.error("No team name column found in player_df")
            return {}
    
    # Map team stats columns to expected names
    column_mapping = {
        'Passing Att': 'Passing Att',
        'Passing Yds': 'Passing Yds',
        'Rushing Att': 'Rushing Att',
        'Rushing Yds': 'Rushing Yds',
        'PF': 'PF',
        'Yds': 'Yds'
    }
    
    # Rename columns if needed
    for old_col, new_col in column_mapping.items():
        if old_col in team_df.columns and new_col not in team_df.columns:
            team_df[new_col] = team_df[old_col]
            logger.info(f"Renamed {old_col} to {new_col}")
    
    # Check for required columns
    required_team_cols = ['Team_std', 'PF', 'Yds', 'Passing Att', 'Passing Yds', 'Rushing Att', 'Rushing Yds']
    missing_team_cols = [col for col in required_team_cols if col not in team_df.columns]
    
    if missing_team_cols:
        logger.error(f"Missing required team columns: {missing_team_cols}")
        return {}
    
    results = {}
    
    # 1. Create team performance tiers
    team_metrics = ['PF', 'Yds', 'Passing Att', 'Passing Yds', 'Rushing Att', 'Rushing Yds']
    
    # Categorize teams into tiers for each metric
    for metric in team_metrics:
        team_df[f"{metric}_Tier"] = pd.qcut(
            team_df[metric], 
            q=3, 
            labels=['Low', 'Medium', 'High']
        )
    
    # 2. Calculate correlation between team and player metrics
    positions = ['QB', 'RB', 'WR', 'TE']
    position_correlations = []
    
    for pos in positions:
        pos_df = player_df[player_df['FantPos'] == pos].copy()
        
        if pos_df.empty:
            logger.warning(f"No {pos} players found, skipping correlation analysis")
            continue
        
        # Log available columns for this position
        logger.info(f"Columns available for {pos}: {pos_df.columns.tolist()}")
        
        # Merge with team data
        pos_team_df = pd.merge(pos_df, team_df, on='Team_std', how='inner')
        
        if pos_team_df.empty:
            logger.warning(f"Failed to match {pos} players with team data")
            logger.info(f"Unique teams in player_df: {pos_df['Team_std'].unique()}")
            logger.info(f"Unique teams in team_df: {team_df['Team_std'].unique()}")
            continue
        
        # Define correlation pairs based on position
        correlation_pairs = []
        
        if pos == 'QB':
            correlation_pairs = [
                ('Passing Att', 'Half_PPR'),
                ('Passing Yds', 'Half_PPR'),
                ('PF', 'Half_PPR'),
                ('Yds', 'Passing Yds')
            ]
        elif pos == 'RB':
            correlation_pairs = [
                ('Rushing Att', 'Half_PPR'),
                ('Rushing Yds', 'Half_PPR'),
                ('PF', 'Half_PPR'),
                ('Passing Att', 'Receiving Rec')
            ]
        elif pos == 'WR':
            correlation_pairs = [
                ('Passing Att', 'Half_PPR'),
                ('Passing Yds', 'Half_PPR'),
                ('PF', 'Half_PPR'),
                ('Yds', 'Receiving Yds')
            ]
        elif pos == 'TE':
            correlation_pairs = [
                ('Passing Att', 'Half_PPR'),
                ('PF', 'Half_PPR'),
                ('Passing Yds', 'Receiving Yds')
            ]
        
        # Calculate correlations
        for team_metric, player_metric in correlation_pairs:
            if team_metric in pos_team_df.columns and player_metric in pos_team_df.columns:
                # Check for sufficient data
                valid_data = pos_team_df[[team_metric, player_metric]].dropna()
                
                if len(valid_data) >= 5:  # Require at least 5 data points
                    corr, p_value = pearsonr(valid_data[team_metric], valid_data[player_metric])
                    
                    correlation_data = {
                        'Position': pos,
                        'Team_Metric': team_metric,
                        'Player_Metric': player_metric,
                        'Correlation': corr,
                        'P_Value': p_value,
                        'Data_Points': len(valid_data)
                    }
                    position_correlations.append(correlation_data)
                    logger.info(f"Calculated correlation for {pos}: {team_metric} vs {player_metric}")
    
    # Create and log the correlation DataFrame
    if position_correlations:
        results['position_team_correlations'] = pd.DataFrame(position_correlations)
        logger.info(f"Created correlation DataFrame with shape: {results['position_team_correlations'].shape}")
        logger.info(f"Correlation DataFrame columns: {results['position_team_correlations'].columns.tolist()}")
    else:
        logger.warning("No correlations were calculated")
        results['position_team_correlations'] = pd.DataFrame()
    
    # 3. Group player performance by team tiers
    team_tier_performance = []
    
    for pos in positions:
        pos_df = player_df[player_df['FantPos'] == pos].copy()
        if pos_df.empty:
            continue
        
        # Identify relevant team metrics for this position
        relevant_metrics = {
            'QB': ['PF', 'Passing Att', 'Passing Yds'],
            'RB': ['PF', 'Rushing Att', 'Rushing Yds', 'Passing Att'],
            'WR': ['PF', 'Passing Att', 'Passing Yds'],
            'TE': ['PF', 'Passing Att', 'Passing Yds']
        }.get(pos, ['PF'])
        
        for metric in relevant_metrics:
            tier_col = f"{metric}_Tier"
            if tier_col not in team_df.columns:
                continue
                
            # Merge player data with team tiers
            merged_df = pd.merge(pos_df, team_df[['Team_std', tier_col]], on='Team_std', how='inner')
            
            if merged_df.empty:
                continue
                
            # Calculate average points by tier
            tier_stats = merged_df.groupby(tier_col).agg({
                'Half_PPR': ['mean', 'std', 'count']
            }).reset_index()
            
            # Flatten the multiindex columns
            tier_stats.columns = [
                '_'.join(col).strip('_') if isinstance(col, tuple) else col 
                for col in tier_stats.columns.values
            ]
            
            # Add position and metric info
            tier_stats['Position'] = pos
            tier_stats['Team_Metric'] = metric
            
            team_tier_performance.append(tier_stats)
    
    if team_tier_performance:
        results['team_tier_performance'] = pd.concat(team_tier_performance, ignore_index=True)
        logger.info(f"Created team tier performance DataFrame with shape: {results['team_tier_performance'].shape}")
    else:
        logger.warning("No team tier performance data was calculated")
        results['team_tier_performance'] = pd.DataFrame()
    
    logger.info("Team performance correlation analysis completed")
    return results

def analyze_opportunity_share(player_df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze player opportunity share within their teams and its impact on fantasy production.
    
    Args:
        player_df: Player performance dataframe
        team_df: Team stats dataframe
        
    Returns:
        pd.DataFrame: Dataframe with opportunity share analysis
    """
    logger.info("Analyzing opportunity share")
    
    # Ensure required columns exist
    required_player_cols = ['Player', 'Team_std', 'FantPos', 'Half_PPR']
    required_team_cols = ['Team_std', 'Passing Att', 'Rushing Att']
    
    missing_player_cols = [col for col in required_player_cols if col not in player_df.columns]
    missing_team_cols = [col for col in required_team_cols if col not in team_df.columns]
    
    if missing_player_cols or missing_team_cols:
        logger.error(f"Missing required columns - Player: {missing_player_cols}, Team: {missing_team_cols}")
        return pd.DataFrame()
    
    # Create a copy to avoid modifying the original
    result_df = player_df.copy()
    
    # Merge with team data
    result_df = pd.merge(result_df, team_df, on='Team_std', how='left')
    
    # Calculate opportunity share by position
    positions = ['QB', 'RB', 'WR', 'TE']
    
    for pos in positions:
        pos_mask = result_df['FantPos'] == pos
        
        if pos == 'QB':
            if all(col in result_df.columns for col in ['Passing Att', 'Passing Att_y']):
                # Calculate passing attempt share
                result_df.loc[pos_mask, 'Pass_Att_Share'] = (
                    result_df.loc[pos_mask, 'Passing Att_x'] / result_df.loc[pos_mask, 'Passing Att_y']
                )
            
            if all(col in result_df.columns for col in ['Rushing Att', 'Rushing Att_y']):
                # Calculate rushing attempt share for QBs
                result_df.loc[pos_mask, 'Rush_Att_Share'] = (
                    result_df.loc[pos_mask, 'Rushing Att_x'] / result_df.loc[pos_mask, 'Rushing Att_y']
                )
        
        elif pos == 'RB':
            if all(col in result_df.columns for col in ['Rushing Att', 'Rushing Att_y']):
                # Calculate rushing attempt share
                result_df.loc[pos_mask, 'Rush_Att_Share'] = (
                    result_df.loc[pos_mask, 'Rushing Att_x'] / result_df.loc[pos_mask, 'Rushing Att_y']
                )
            
            if all(col in result_df.columns for col in ['Receiving Tgt', 'Passing Att_y']):
                # Calculate target share
                result_df.loc[pos_mask, 'Target_Share'] = (
                    result_df.loc[pos_mask, 'Receiving Tgt'] / result_df.loc[pos_mask, 'Passing Att_y']
                )
        
        elif pos in ['WR', 'TE']:
            if all(col in result_df.columns for col in ['Receiving Tgt', 'Passing Att_y']):
                # Calculate target share
                result_df.loc[pos_mask, 'Target_Share'] = (
                    result_df.loc[pos_mask, 'Receiving Tgt'] / result_df.loc[pos_mask, 'Passing Att_y']
                )
    
    # Replace inf/NaN values
    share_cols = [col for col in result_df.columns if 'Share' in col]
    for col in share_cols:
        result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate fantasy points per opportunity
    if all(col in result_df.columns for col in ['Rushing Att_x', 'Receiving Tgt']):
        result_df['Total_Opportunities'] = result_df['Rushing Att_x'].fillna(0) + result_df['Receiving Tgt'].fillna(0)
        result_df['Points_Per_Opportunity'] = result_df['Half_PPR'] / result_df['Total_Opportunities'].replace(0, np.nan)
        result_df['Points_Per_Opportunity'] = result_df['Points_Per_Opportunity'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Create efficiency vs. volume quadrants
    for pos in positions:
        pos_df = result_df[result_df['FantPos'] == pos].copy()
        if len(pos_df) < 5:
            continue
        
        # Determine the opportunity share column to use
        if pos == 'QB':
            share_col = 'Pass_Att_Share'
        elif pos == 'RB':
            share_col = 'Rush_Att_Share'
        else:
            share_col = 'Target_Share'
        
        if share_col not in pos_df.columns or 'Points_Per_Opportunity' not in pos_df.columns:
            continue
        
        # Calculate medians for the position
        opp_median = pos_df[share_col].median()
        eff_median = pos_df['Points_Per_Opportunity'].median()
        
        # Assign quadrants
        pos_mask = result_df['FantPos'] == pos
        
        # High volume, high efficiency (stars)
        result_df.loc[
            pos_mask & 
            (result_df[share_col] >= opp_median) & 
            (result_df['Points_Per_Opportunity'] >= eff_median),
            'Efficiency_Quadrant'
        ] = 'High Vol, High Eff'
        
        # High volume, low efficiency (workhorses)
        result_df.loc[
            pos_mask & 
            (result_df[share_col] >= opp_median) & 
            (result_df['Points_Per_Opportunity'] < eff_median),
            'Efficiency_Quadrant'
        ] = 'High Vol, Low Eff'
        
        # Low volume, high efficiency (efficient backups)
        result_df.loc[
            pos_mask & 
            (result_df[share_col] < opp_median) & 
            (result_df['Points_Per_Opportunity'] >= eff_median),
            'Efficiency_Quadrant'
        ] = 'Low Vol, High Eff'
        
        # Low volume, low efficiency (bench players)
        result_df.loc[
            pos_mask & 
            (result_df[share_col] < opp_median) & 
            (result_df['Points_Per_Opportunity'] < eff_median),
            'Efficiency_Quadrant'
        ] = 'Low Vol, Low Eff'
    
    logger.info("Opportunity share analysis completed")
    return result_df

def analyze_offensive_line_impact(player_df: pd.DataFrame, line_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze the impact of offensive line performance on QB and RB production.
    
    Args:
        player_df: Player performance dataframe
        line_df: Offensive line performance dataframe
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of dataframes with offensive line impact analysis
    """
    logger.info("Analyzing offensive line impact")
    
    # Ensure team names are standardized
    if 'Team_std' not in line_df.columns and 'team_name' in line_df.columns:
        line_df['Team_std'] = line_df['team_name']
    
    # Check for required columns
    required_line_cols = ['Team_std', 'pbe', 'pressures_allowed', 'sacks_allowed']
    missing_line_cols = [col for col in required_line_cols if col not in line_df.columns]
    
    if missing_line_cols:
        logger.error(f"Missing required offensive line columns: {missing_line_cols}")
        return {}
    
    results = {}
    
    # Create offensive line tiers
    line_metrics = ['pbe', 'pressures_allowed', 'sacks_allowed']
    
    for metric in line_metrics:
        # For PBE, higher is better; for pressures and sacks, lower is better
        ascending = metric == 'pbe'
        
        line_df[f"{metric}_Tier"] = pd.qcut(
            line_df[metric],
            q=3,
            labels=['Low', 'Medium', 'High'] if ascending else ['High', 'Medium', 'Low']
        )
    
    # Analyze QB performance by offensive line tiers
    qb_df = player_df[player_df['FantPos'] == 'QB'].copy()
    if not qb_df.empty:
        # Merge with line data
        qb_line_df = pd.merge(qb_df, line_df, on='Team_std', how='inner')
        
        if not qb_line_df.empty:
            # Calculate QB performance by line tier
            qb_tier_stats = []
            
            for metric in line_metrics:
                tier_col = f"{metric}_Tier"
                
                tier_perf = qb_line_df.groupby(tier_col).agg({
                    'Half_PPR': ['mean', 'std', 'count'],
                    'Passing Yds': 'mean',
                    'Passing TD': 'mean',
                    'Passing Int': 'mean'
                }).reset_index()
                
                # Flatten the multiindex columns
                tier_perf.columns = [
                    '_'.join(col).strip('_') if isinstance(col, tuple) else col 
                    for col in tier_perf.columns.values
                ]
                
                # Add metric info
                tier_perf['Line_Metric'] = metric
                
                qb_tier_stats.append(tier_perf)
            
            if qb_tier_stats:
                results['qb_line_tier_stats'] = pd.concat(qb_tier_stats, ignore_index=True)
            
            # Calculate correlations between line metrics and QB performance
            qb_line_corr = []
            
            for metric in line_metrics:
                if metric in qb_line_df.columns and 'Half_PPR' in qb_line_df.columns:
                    valid_data = qb_line_df[[metric, 'Half_PPR']].dropna()
                    
                    if len(valid_data) >= 5:
                        corr, p_value = pearsonr(valid_data[metric], valid_data['Half_PPR'])
                        
                        correlation_data = {
                            'Line_Metric': metric,
                            'Player_Metric': 'Half_PPR',
                            'Correlation': corr,
                            'P_Value': p_value,
                            'Data_Points': len(valid_data)
                        }
                        qb_line_corr.append(correlation_data)
            
            if qb_line_corr:
                results['qb_line_correlations'] = pd.DataFrame(qb_line_corr)
    
    # Analyze RB performance by offensive line tiers
    rb_df = player_df[player_df['FantPos'] == 'RB'].copy()
    if not rb_df.empty:
        # Merge with line data
        rb_line_df = pd.merge(rb_df, line_df, on='Team_std', how='inner')
        
        if not rb_line_df.empty:
            # Calculate RB performance by line tier
            rb_tier_stats = []
            
            for metric in line_metrics:
                tier_col = f"{metric}_Tier"
                
                tier_perf = rb_line_df.groupby(tier_col).agg({
                    'Half_PPR': ['mean', 'std', 'count'],
                    'Rushing Yds': 'mean',
                    'Rushing TD': 'mean',
                    'Rushing Y/A': 'mean'
                }).reset_index()
                
                # Flatten the multiindex columns
                tier_perf.columns = [
                    '_'.join(col).strip('_') if isinstance(col, tuple) else col 
                    for col in tier_perf.columns.values
                ]
                
                # Add metric info
                tier_perf['Line_Metric'] = metric
                
                rb_tier_stats.append(tier_perf)
            
            if rb_tier_stats:
                results['rb_line_tier_stats'] = pd.concat(rb_tier_stats, ignore_index=True)
            
            # Calculate correlations between line metrics and RB performance
            rb_line_corr = []
            
            for metric in line_metrics:
                if metric in rb_line_df.columns:
                    for perf_metric in ['Half_PPR', 'Rushing Yds', 'Rushing Y/A']:
                        if perf_metric in rb_line_df.columns:
                            valid_data = rb_line_df[[metric, perf_metric]].dropna()
                            
                            if len(valid_data) >= 5:
                                corr, p_value = pearsonr(valid_data[metric], valid_data[perf_metric])
                                
                                correlation_data = {
                                    'Line_Metric': metric,
                                    'Player_Metric': perf_metric,
                                    'Correlation': corr,
                                    'P_Value': p_value,
                                    'Data_Points': len(valid_data)
                                }
                                rb_line_corr.append(correlation_data)
            
            if rb_line_corr:
                results['rb_line_correlations'] = pd.DataFrame(rb_line_corr)
    
    logger.info("Offensive line impact analysis completed")
    return results