import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def identify_player_tiers(df: pd.DataFrame, min_clusters: int = 2, max_clusters: int = 8) -> Dict[str, pd.DataFrame]:
    """
    Identify natural performance tiers for each position using clustering.
    
    Args:
        df: Player performance dataframe
        min_clusters: Minimum number of clusters to consider
        max_clusters: Maximum number of clusters to consider
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of dataframes with tier assignments by position
    """
    logger.info("Identifying player performance tiers")
    
    positions = ['QB', 'RB', 'WR', 'TE']
    results = {}
    
    for pos in positions:
        pos_df = df[df['FantPos'] == pos].copy()
        if len(pos_df) < min_clusters * 2:
            logger.warning(f"Not enough {pos} players for meaningful clustering, skipping")
            continue
        
        # Sort by points
        pos_df = pos_df.sort_values('Half_PPR', ascending=False).reset_index(drop=True)
        
        # Prepare data for clustering
        X = pos_df[['Half_PPR']].values
        
        # Find optimal number of clusters using elbow method
        inertias = []
        for k in range(min_clusters, min(max_clusters + 1, len(pos_df) // 2 + 1)):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point - where adding more clusters doesn't help much
        optimal_k = min_clusters
        if len(inertias) > 1:
            diffs = np.diff(inertias)
            diffs_of_diffs = np.diff(diffs)
            if len(diffs_of_diffs) > 0:
                # Find where the rate of improvement slows down
                elbow_idx = np.argmax(diffs_of_diffs) + 1 if np.any(diffs_of_diffs > 0) else 0
                optimal_k = min_clusters + elbow_idx
        
        # Apply KMeans with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Add tier assignments to dataframe
        pos_df['Tier_Cluster'] = clusters
        
        # Map cluster numbers to tiers (Tier 1 should be the best)
        # Get the average points per cluster
        cluster_means = pos_df.groupby('Tier_Cluster')['Half_PPR'].mean().reset_index()
        cluster_means = cluster_means.sort_values('Half_PPR', ascending=False).reset_index(drop=True)
        
        # Create tier mapping
        tier_mapping = {row['Tier_Cluster']: f"Tier {i+1}" for i, (_, row) in enumerate(cluster_means.iterrows())}
        pos_df['Tier'] = pos_df['Tier_Cluster'].map(tier_mapping)
        
        # Calculate tier stats
        tier_stats = pos_df.groupby('Tier').agg({
            'Half_PPR': ['mean', 'std', 'min', 'max', 'count']
        }).reset_index()
        tier_stats.columns = ['Tier', 'Avg_Points', 'Std_Dev', 'Min_Points', 'Max_Points', 'Player_Count']
        
        # Add example players to tier stats
        tier_examples = []
        for tier in tier_stats['Tier']:
            tier_players = pos_df[pos_df['Tier'] == tier].sort_values('Half_PPR', ascending=False)
            examples = tier_players.head(3)['Player'].tolist() if len(tier_players) >= 3 else tier_players['Player'].tolist()
            tier_examples.append(', '.join(examples))
        
        tier_stats['Example_Players'] = tier_examples
        
        # Store results
        results[f"{pos}_tiers"] = pos_df
        results[f"{pos}_tier_stats"] = tier_stats
    
    logger.info("Player tier identification completed")
    return results

def identify_player_archetypes(df: pd.DataFrame, position_features: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    """
    Identify player archetypes within each position based on playing style.
    
    Args:
        df: Player performance dataframe with advanced metrics
        position_features: Dictionary mapping positions to feature lists for clustering
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of dataframes with archetype assignments by position
    """
    logger.info("Identifying player archetypes")
    
    positions = list(position_features.keys())
    results = {}
    
    for pos in positions:
        if pos not in position_features or not position_features[pos]:
            logger.warning(f"No features defined for {pos}, skipping archetype identification")
            continue
        
        # Filter position and keep only players with all required features
        pos_df = df[df['FantPos'] == pos].copy()
        features = position_features[pos]
        
        # Check for required columns
        missing_cols = [col for col in features if col not in pos_df.columns]
        if missing_cols:
            logger.warning(f"Missing features for {pos} archetype identification: {missing_cols}")
            # Use available features
            features = [col for col in features if col in pos_df.columns]
            if not features:
                logger.error(f"No available features for {pos}, skipping")
                continue
        
        # Filter players with all features available
        pos_df = pos_df.dropna(subset=features)
        if len(pos_df) < 5:
            logger.warning(f"Not enough {pos} players with complete feature data, skipping")
            continue
        
        # Standardize features
        X = pos_df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction if we have many features
        if len(features) > 2:
            n_components = min(len(features), 3)  # Use 2-3 components for easier visualization
            pca = PCA(n_components=n_components, random_state=42)
            X_reduced = pca.fit_transform(X_scaled)
            
            # Store component information
            component_df = pd.DataFrame(
                pca.components_,
                columns=features,
                index=[f'Component {i+1}' for i in range(n_components)]
            )
            results[f"{pos}_pca_components"] = component_df
            
            # Store explained variance
            explained_variance = pd.DataFrame({
                'Component': [f'Component {i+1}' for i in range(n_components)],
                'Explained_Variance': pca.explained_variance_ratio_,
                'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
            })
            results[f"{pos}_pca_variance"] = explained_variance
        else:
            X_reduced = X_scaled
        
        # Determine number of clusters (archetypes)
        # Start with a reasonable number based on position
        n_clusters = {'QB': 3, 'RB': 4, 'WR': 4, 'TE': 3}.get(pos, 3)
        
        # Ensure we don't try to create more clusters than samples
        n_clusters = min(n_clusters, len(pos_df) // 3)
        if n_clusters < 2:
            n_clusters = 2  # Minimum 2 clusters
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_reduced)
        
        # Add archetype assignments to dataframe
        pos_df['Archetype_Cluster'] = clusters
        
        # Create archetype labels based on feature importance
        archetype_centers = {}
        for cluster in range(n_clusters):
            cluster_points = X_scaled[clusters == cluster]
            center = np.mean(cluster_points, axis=0)
            archetype_centers[cluster] = center
        
        # Create descriptive labels for archetypes
        archetype_labels = {}
        for cluster, center in archetype_centers.items():
            # Find top 2 defining features (highest absolute values)
            feature_importance = [(features[i], center[i]) for i in range(len(features))]
            sorted_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)
            
            top_features = sorted_features[:2]
            descriptors = []
            for feature, value in top_features:
                if value > 0.5:
                    descriptors.append(f"High {feature.replace('_', ' ')}")
                elif value < -0.5:
                    descriptors.append(f"Low {feature.replace('_', ' ')}")
            
            if descriptors:
                label = f"Archetype {cluster+1}: {' + '.join(descriptors)}"
            else:
                label = f"Archetype {cluster+1}: Balanced"
                
            archetype_labels[cluster] = label
        
        # Apply labels
        pos_df['Archetype'] = pos_df['Archetype_Cluster'].map(archetype_labels)
        
        # Calculate archetype stats
        archetype_stats = pos_df.groupby('Archetype').agg({
            'Half_PPR': ['mean', 'std', 'min', 'max', 'count']
        }).reset_index()
        archetype_stats.columns = ['Archetype', 'Avg_Points', 'Std_Dev', 'Min_Points', 'Max_Points', 'Player_Count']
        
        # Add example players to archetype stats
        archetype_examples = []
        for archetype in archetype_stats['Archetype']:
            arch_players = pos_df[pos_df['Archetype'] == archetype].sort_values('Half_PPR', ascending=False)
            examples = arch_players.head(3)['Player'].tolist() if len(arch_players) >= 3 else arch_players['Player'].tolist()
            archetype_examples.append(', '.join(examples))
        
        archetype_stats['Example_Players'] = archetype_examples
        
        # Calculate feature means for each archetype for interpretation
        feature_means = pos_df.groupby('Archetype')[features].mean().reset_index()
        
        # Store results
        results[f"{pos}_archetypes"] = pos_df
        results[f"{pos}_archetype_stats"] = archetype_stats
        results[f"{pos}_archetype_features"] = feature_means
    
    logger.info("Player archetype identification completed")
    return results

def analyze_tiers_vs_expectations(tier_results: Dict[str, pd.DataFrame], df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze how different tiers performed relative to pre-season expectations.
    
    Args:
        tier_results: Dictionary of dataframes with tier assignments
        df: Player performance dataframe with pre-season rankings
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of dataframes with tier vs. expectations analysis
    """
    logger.info("Analyzing tiers vs. expectations")
    
    positions = ['QB', 'RB', 'WR', 'TE']
    results = {}
    
    for pos in positions:
        tier_key = f"{pos}_tiers"
        if tier_key not in tier_results:
            logger.warning(f"No tier data found for {pos}, skipping expectations analysis")
            continue
        
        tier_df = tier_results[tier_key]
        
        # Ensure we have ADP data
        if 'ADP' not in tier_df.columns:
            # Try to merge with original dataframe
            tier_df = pd.merge(
                tier_df,
                df[['Player', 'Team_std', 'ADP']],
                on=['Player', 'Team_std'],
                how='left'
            )
        
        if 'ADP' not in tier_df.columns or tier_df['ADP'].isnull().all():
            logger.warning(f"No ADP data available for {pos}, skipping expectations analysis")
            continue
        
        # Analyze expectations vs. results by tier
        tier_expectations = tier_df.groupby('Tier').agg({
            'ADP': 'mean',
            'Half_PPR': 'mean',
            'Player': 'count'
        }).reset_index()
        tier_expectations.columns = ['Tier', 'Avg_ADP', 'Avg_Points', 'Player_Count']
        
        # Calculate expected draft round
        team_count = 12  # Default to 12-team league
        tier_expectations['Expected_Round'] = np.ceil(tier_expectations['Avg_ADP'] / team_count)
        
        # Calculate success metrics
        if 'PFF Points' in tier_df.columns:
            # Calculate over/under performance relative to projections
            tier_df['Performance_vs_Projection'] = tier_df['Half_PPR'] / tier_df['PFF Points'] - 1
            
            # Calculate success rate (% of players beating projections)
            tier_success = tier_df.groupby('Tier').apply(
                lambda x: (x['Performance_vs_Projection'] > 0).mean()
            ).reset_index()
            tier_success.columns = ['Tier', 'Success_Rate']
            
            # Merge success metrics
            tier_expectations = pd.merge(tier_expectations, tier_success, on='Tier', how='left')
        
        # Calculate ROI (return on investment)
        tier_expectations['Points_per_ADP'] = tier_expectations['Avg_Points'] / tier_expectations['Avg_ADP']
        
        # Sort by tier number
        tier_expectations['Tier_Num'] = tier_expectations['Tier'].str.extract(r'Tier (\d+)').astype(int)
        tier_expectations = tier_expectations.sort_values('Tier_Num').drop('Tier_Num', axis=1)
        
        # Store results
        results[f"{pos}_tier_expectations"] = tier_expectations
    
    # Create a cross-position tier ROI analysis
    all_tier_data = []
    for pos in positions:
        key = f"{pos}_tier_expectations"
        if key in results:
            pos_data = results[key].copy()
            pos_data['Position'] = pos
            all_tier_data.append(pos_data)
    
    if all_tier_data:
        all_tier_df = pd.concat(all_tier_data, ignore_index=True)
        results['all_tier_expectations'] = all_tier_df
    
    logger.info("Tier vs. expectations analysis completed")
    return results