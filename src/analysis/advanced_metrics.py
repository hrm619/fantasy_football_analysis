import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from scipy.stats import pearsonr

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_efficiency_metrics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze advanced efficiency metrics by position and identify potential values.
    
    Args:
        df: Player performance dataframe with advanced metrics
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of dataframes with efficiency analysis
    """
    logger.info("Analyzing efficiency metrics")
    
    results = {}
    
    # Define position-specific efficiency metrics
    efficiency_metrics = {
        'QB': {
            'Air_Yards_Per_Att': ['avg_depth_of_target'],
            'TD_Rate': ['btt_rate', 'completion_percent', 'qb_rating'],
            'Int_Avoidance': ['twp_rate', 'interceptions'],
            'Accuracy': ['accuracy_percent', 'completion_percent']
        },
        'RB': {
            'YAC_Ability': ['yards_after_contact', 'yco_attempt'],
            'Tackle_Breaking': ['avoided_tackles', 'elusive_rating'],
            'Receiving_Efficiency': ['yprr', 'catch_rate', 'received_yards_per_reception']
        },
        'WR': {
            'Route_Efficiency': ['yprr'],
            'Downfield_Ability': ['avg_depth_of_target'],
            'YAC_Ability': ['yards_after_catch_per_reception'],
            'Contested_Catch': ['contested_catch_rate']
        },
        'TE': {
            'Route_Efficiency': ['yprr', 'route_rate'],
            'Receiving_Efficiency': ['catch_rate', 'yards_per_reception'],
            'Red_Zone_Efficiency': ['touchdowns', 'targeted_qb_rating']
        }
    }
    
    # Check which metrics are available in the dataframe
    available_metrics = {}
    
    for pos, metric_dict in efficiency_metrics.items():
        available_metrics[pos] = {}
        for category, metrics in metric_dict.items():
            available_metrics[pos][category] = [
                metric for metric in metrics if metric in df.columns
            ]
    
    # Calculate correlations between efficiency metrics and fantasy points
    positions = ['QB', 'RB', 'WR', 'TE']
    metric_correlations = []
    
    for pos in positions:
        pos_df = df[df['FantPos'] == pos].copy()
        if pos_df.empty:
            logger.warning(f"No {pos} players found, skipping efficiency analysis")
            continue
        
        # Get available metrics for this position
        pos_metrics = []
        for category, metrics in available_metrics[pos].items():
            pos_metrics.extend(metrics)
        
        # Calculate correlations with fantasy points
        for metric in pos_metrics:
            if metric in pos_df.columns:
                valid_data = pos_df[['Half_PPR', metric]].dropna()
                
                if len(valid_data) >= 5:  # Require at least 5 data points
                    corr, p_value = pearsonr(valid_data['Half_PPR'], valid_data[metric])
                    
                    metric_corr = {
                        'Position': pos,
                        'Metric': metric,
                        'Correlation': corr,
                        'P_Value': p_value,
                        'Significant': p_value < 0.05,
                        'Sample_Size': len(valid_data)
                    }
                    metric_correlations.append(metric_corr)
    
    if metric_correlations:
        results['efficiency_correlations'] = pd.DataFrame(metric_correlations)
    
    # Identify efficiency outliers (potential regression/improvement candidates)
    efficiency_outliers = []
    
    for pos in positions:
        pos_df = df[df['FantPos'] == pos].copy()
        if pos_df.empty:
            continue
        
        # Get key efficiency metrics for this position
        key_metrics = []
        for metrics in available_metrics[pos].values():
            key_metrics.extend(metrics)
        
        # Filter to metrics with strong correlations
        if 'efficiency_correlations' in results:
            corr_df = results['efficiency_correlations']
            strong_metrics = corr_df[
                (corr_df['Position'] == pos) & 
                (corr_df['Significant']) & 
                (abs(corr_df['Correlation']) > 0.3)  # Moderate to strong correlation
            ]['Metric'].tolist()
            
            key_metrics = [m for m in key_metrics if m in strong_metrics]
        
        # Use available metrics
        key_metrics = [m for m in key_metrics if m in pos_df.columns]
        
        if not key_metrics:
            logger.warning(f"No key efficiency metrics available for {pos}")
            continue
        
        # Calculate z-scores for each metric
        for metric in key_metrics:
            z_col = f"{metric}_Z"
            pos_df[z_col] = (pos_df[metric] - pos_df[metric].mean()) / pos_df[metric].std()
        
        # Calculate composite efficiency score
        z_cols = [f"{metric}_Z" for metric in key_metrics]
        if z_cols:
            pos_df['Efficiency_Score'] = pos_df[z_cols].mean(axis=1)
            
            # Calculate fantasy points per efficiency score
            pos_df['Fantasy_Points_Per_Efficiency'] = pos_df['Half_PPR'] / pos_df['Efficiency_Score'].abs()
            pos_df['Fantasy_Points_Per_Efficiency'] = pos_df['Fantasy_Points_Per_Efficiency'].replace(
                [np.inf, -np.inf], np.nan
            ).fillna(pos_df['Half_PPR'])
            
            # Identify outliers
            threshold = 1.5  # z-score threshold for outliers
            
            # Positive outliers (highly efficient)
            high_eff = pos_df[pos_df['Efficiency_Score'] > threshold].copy()
            high_eff['Outlier_Type'] = 'High Efficiency'
            high_eff['Regression_Risk'] = high_eff['Efficiency_Score'] > 2.0  # Very high efficiency might regress
            
            # Negative outliers (inefficient)
            low_eff = pos_df[pos_df['Efficiency_Score'] < -threshold].copy()
            low_eff['Outlier_Type'] = 'Low Efficiency'
            low_eff['Improvement_Potential'] = low_eff['Efficiency_Score'] < -2.0  # Very low efficiency might improve
            
            # Combine outliers
            outliers = pd.concat([high_eff, low_eff])
            
            if not outliers.empty:
                outliers['Position'] = pos
                efficiency_outliers.append(outliers[['Player', 'Position', 'Team_std', 'Half_PPR', 
                                                   'Efficiency_Score', 'Outlier_Type', 'Fantasy_Points_Per_Efficiency',
                                                   'Regression_Risk', 'Improvement_Potential']])
    
    if efficiency_outliers:
        results['efficiency_outliers'] = pd.concat(efficiency_outliers, ignore_index=True)
    
    # Calculate composite efficiency scores for all players
    composite_scores = []
    
    for pos in positions:
        pos_df = df[df['FantPos'] == pos].copy()
        if pos_df.empty:
            continue
        
        # Get key metrics for this position
        key_metrics = []
        for category, metrics in available_metrics[pos].items():
            if metrics:
                # For each category, use the metric with the strongest correlation if available
                if 'efficiency_correlations' in results:
                    corr_df = results['efficiency_correlations']
                    category_corrs = corr_df[
                        (corr_df['Position'] == pos) & 
                        (corr_df['Metric'].isin(metrics))
                    ]
                    
                    if not category_corrs.empty:
                        # Get the metric with the strongest absolute correlation
                        best_metric = category_corrs.loc[
                            category_corrs['Correlation'].abs().idxmax()
                        ]['Metric']
                        key_metrics.append(best_metric)
                    else:
                        # If no correlation data, use the first available metric
                        key_metrics.append(metrics[0])
                else:
                    # If no correlation data, use the first available metric
                    key_metrics.append(metrics[0])
        
        # Filter to available metrics
        key_metrics = [m for m in key_metrics if m in pos_df.columns]
        
        if not key_metrics:
            continue
        
        # Calculate z-scores
        for metric in key_metrics:
            z_col = f"{metric}_Z"
            pos_df[z_col] = (pos_df[metric] - pos_df[metric].mean()) / pos_df[metric].std()
        
        # Calculate composite score
        z_cols = [f"{metric}_Z" for metric in key_metrics]
        if z_cols:
            pos_df['Composite_Efficiency'] = pos_df[z_cols].mean(axis=1)
            
            # Add metric weights used
            metric_weights = {metric: 1/len(key_metrics) for metric in key_metrics}
            pos_df['Efficiency_Metrics_Used'] = str(metric_weights)
            
            # Add to results
            pos_df['Position'] = pos
            composite_scores.append(pos_df[['Player', 'Position', 'Team_std', 'Half_PPR', 
                                           'Composite_Efficiency', 'Efficiency_Metrics_Used']])
    
    if composite_scores:
        results['composite_efficiency_scores'] = pd.concat(composite_scores, ignore_index=True)
    
    logger.info("Efficiency metrics analysis completed")
    return results

def analyze_pff_grade_correlations(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze correlations between PFF grades and fantasy performance.
    
    Args:
        df: Player performance dataframe with PFF grades
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of dataframes with PFF grade analysis
    """
    logger.info("Analyzing PFF grade correlations")
    
    results = {}
    
    # Define PFF grade columns by position
    pff_grades = {
        'QB': ['grades_offense', 'grades_pass', 'grades_run', 'grades_hands_fumble'],
        'RB': ['grades_offense', 'grades_run', 'grades_hands_fumble', 'grades_pass_route', 'grades_pass_block'],
        'WR': ['grades_offense', 'grades_pass_route', 'grades_hands_drop', 'grades_hands_fumble'],
        'TE': ['grades_offense', 'grades_pass_route', 'grades_pass_block', 'grades_hands_drop', 'grades_hands_fumble']
    }
    
    # Calculate correlations between PFF grades and fantasy points
    positions = ['QB', 'RB', 'WR', 'TE']
    grade_correlations = []
    
    for pos in positions:
        pos_df = df[df['FantPos'] == pos].copy()
        if pos_df.empty:
            logger.warning(f"No {pos} players found, skipping PFF grade analysis")
            continue
        
        # Get available grade columns
        grade_cols = [col for col in pff_grades.get(pos, []) if col in pos_df.columns]
        
        if not grade_cols:
            logger.warning(f"No PFF grade columns found for {pos}")
            continue
        
        # Calculate correlations
        for grade_col in grade_cols:
            valid_data = pos_df[['Half_PPR', grade_col]].dropna()
            
            if len(valid_data) >= 5:  # Require at least 5 data points
                corr, p_value = pearsonr(valid_data['Half_PPR'], valid_data[grade_col])
                
                grade_corr = {
                    'Position': pos,
                    'PFF_Grade': grade_col,
                    'Correlation': corr,
                    'P_Value': p_value,
                    'Significant': p_value < 0.05,
                    'Sample_Size': len(valid_data)
                }
                grade_correlations.append(grade_corr)
    
    if grade_correlations:
        results['pff_grade_correlations'] = pd.DataFrame(grade_correlations)
    
    # Create composite PFF scores
    composite_pff_scores = []
    
    for pos in positions:
        pos_df = df[df['FantPos'] == pos].copy()
        if pos_df.empty:
            continue
        
        # Get available grade columns
        grade_cols = [col for col in pff_grades.get(pos, []) if col in pos_df.columns]
        
        if not grade_cols:
            continue
        
        # Determine weights based on correlations if available
        weights = {}
        if 'pff_grade_correlations' in results:
            corr_df = results['pff_grade_correlations']
            pos_corrs = corr_df[corr_df['Position'] == pos]
            
            for grade_col in grade_cols:
                grade_corr = pos_corrs[pos_corrs['PFF_Grade'] == grade_col]
                if not grade_corr.empty:
                    # Use absolute correlation as weight
                    weights[grade_col] = abs(grade_corr.iloc[0]['Correlation'])
                else:
                    weights[grade_col] = 1.0
        else:
            # Equal weights if correlation data not available
            weights = {col: 1.0 for col in grade_cols}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {col: w/total_weight for col, w in weights.items()}
        
        # Calculate composite score
        pos_df['Composite_PFF_Score'] = 0
        for col, weight in weights.items():
            pos_df['Composite_PFF_Score'] += pos_df[col] * weight
        
        # Add weight information
        pos_df['PFF_Weights_Used'] = str(weights)
        
        # Add to results
        pos_df['Position'] = pos
        composite_pff_scores.append(pos_df[['Player', 'Position', 'Team_std', 'Half_PPR', 
                                           'Composite_PFF_Score', 'PFF_Weights_Used']])
    
    if composite_pff_scores:
        results['composite_pff_scores'] = pd.concat(composite_pff_scores, ignore_index=True)
        
        # Calculate correlation between composite PFF score and fantasy points
        composite_df = results['composite_pff_scores']
        composite_corrs = []
        
        for pos in positions:
            pos_df = composite_df[composite_df['Position'] == pos]
            if len(pos_df) >= 5:
                corr, p_value = pearsonr(pos_df['Half_PPR'], pos_df['Composite_PFF_Score'])
                
                composite_corr = {
                    'Position': pos,
                    'Correlation': corr,
                    'P_Value': p_value,
                    'Significant': p_value < 0.05,
                    'Sample_Size': len(pos_df)
                }
                composite_corrs.append(composite_corr)
        
        if composite_corrs:
            results['composite_pff_correlations'] = pd.DataFrame(composite_corrs)
    
    logger.info("PFF grade correlation analysis completed")
    return results