import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

def plot_top_performers(df: pd.DataFrame, n: int = 10) -> None:
    """
    Plot top performers by position.
    
    Args:
        df: Player performance dataframe
        n: Number of top performers to display
    """
    positions = ['QB', 'RB', 'WR', 'TE']
    
    # Create a 2x2 grid for the positions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # For each position, plot the top n performers
    for i, pos in enumerate(positions):
        pos_df = df[df['FantPos'] == pos].copy()
        
        # Skip if no players for this position
        if pos_df.empty:
            axes[i].text(0.5, 0.5, f"No {pos} data available", 
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[i].transAxes, fontsize=14)
            continue
            
        # Get top n players by Half_PPR
        top_players = pos_df.sort_values('Half_PPR', ascending=False).head(n)
        
        # Create horizontal bar chart
        sns.barplot(x='Half_PPR', y='Player', data=top_players, ax=axes[i],
                   palette='viridis')
        
        # Add data labels
        for j, p in enumerate(top_players['Half_PPR']):
            axes[i].text(p + 2, j, f"{p:.1f}", va='center')
            
        # Add PPG as text
        if 'Half_PPR_PPG' in top_players.columns:
            for j, (_, row) in enumerate(top_players.iterrows()):
                axes[i].text(row['Half_PPR'] * 0.5, j, f"{row['Half_PPR_PPG']:.1f} PPG", 
                          va='center', ha='center', color='white', fontweight='bold')
        
        # Format the plot
        axes[i].set_title(f'Top {n} {pos}s by Half-PPR Points', fontsize=14)
        axes[i].set_xlabel('Half-PPR Points', fontsize=12)
        axes[i].set_ylabel('')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.suptitle('Top Fantasy Performers by Position', fontsize=16)

def plot_position_distributions(df: pd.DataFrame) -> None:
    """
    Plot distributions of fantasy points by position.
    
    Args:
        df: Player performance dataframe
    """
    positions = ['QB', 'RB', 'WR', 'TE']
    
    # Create grid for distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Set up colors for each position
    colors = ['#FF9999', '#99FF99', '#9999FF', '#FFFF99']
    
    # For each position, plot the distribution of points
    for i, pos in enumerate(positions):
        pos_df = df[df['FantPos'] == pos].copy()
        
        # Skip if no players for this position
        if pos_df.empty:
            axes[i].text(0.5, 0.5, f"No {pos} data available", 
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[i].transAxes, fontsize=14)
            continue
        
        # Create KDE plot with histogram
        sns.histplot(pos_df['Half_PPR'], kde=True, ax=axes[i], color=colors[i], bins=20)
        
        # Add position average line
        pos_avg = pos_df['Half_PPR'].mean()
        axes[i].axvline(pos_avg, color='red', linestyle='--', 
                       label=f'Average: {pos_avg:.1f}')
        
        # Add VBD baseline if available
        if 'VORP' in pos_df.columns:
            baseline_player = pos_df[pos_df['VORP'] == 0]
            if not baseline_player.empty:
                baseline = baseline_player['Half_PPR'].values[0]
                axes[i].axvline(baseline, color='green', linestyle='--', 
                               label=f'Baseline: {baseline:.1f}')
        
        # Format the plot
        axes[i].set_title(f'{pos} Fantasy Point Distribution', fontsize=14)
        axes[i].set_xlabel('Half-PPR Points', fontsize=12)
        axes[i].set_ylabel('Count', fontsize=12)
        axes[i].legend()
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.suptitle('Fantasy Point Distributions by Position', fontsize=16)

def plot_expectation_vs_performance(df: pd.DataFrame) -> None:
    """
    Plot comparison between pre-season expectations and actual performance.
    
    Args:
        df: Player performance dataframe with expectation data
    """
    # Check if we have ADP and performance data
    if 'ADP' not in df.columns or 'Half_PPR' not in df.columns:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "Missing ADP or performance data", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
        return
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: ADP vs. Actual Performance Scatter
    positions = df['FantPos'].unique()
    position_colors = {'QB': '#FF9999', 'RB': '#99FF99', 'WR': '#9999FF', 'TE': '#FFFF99'}
    
    for pos in positions:
        pos_df = df[df['FantPos'] == pos]
        ax1.scatter(pos_df['ADP'], pos_df['Half_PPR'], alpha=0.7, 
                   label=pos, s=60, c=position_colors.get(pos, 'gray'))
    
    # Add regression line
    # First, create a clean dataset with no NaN values
    clean_data = df[['ADP', 'Half_PPR']].dropna()
    if len(clean_data) > 1:  # Only fit line if we have enough data points
        adp_range = np.linspace(clean_data['ADP'].min(), clean_data['ADP'].max(), 100)
        z = np.polyfit(clean_data['ADP'], clean_data['Half_PPR'], 1)
        p = np.poly1d(z)
        ax1.plot(adp_range, p(adp_range), "r--", alpha=0.7)
    
    # Add correlation coefficient
    corr = clean_data['ADP'].corr(clean_data['Half_PPR'])
    ax1.text(0.05, 0.95, f"Correlation: {corr:.2f}", transform=ax1.transAxes, 
            fontsize=12, verticalalignment='top')
    
    # Format plot
    ax1.set_title('ADP vs. Actual Fantasy Performance', fontsize=14)
    ax1.set_xlabel('Average Draft Position (ADP)', fontsize=12)
    ax1.set_ylabel('Half-PPR Points', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(title='Position')
    
    # Plot 2: Performance Categories
    if 'Performance_Category' in df.columns:
        # Count players in each category
        cat_counts = df['Performance_Category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        
        # Sort categories in logical order
        order = ['Significant Overperformer', 'Moderate Overperformer', 
                'Met Expectations', 'Moderate Underperformer', 'Significant Underperformer']
        cat_counts['Category'] = pd.Categorical(cat_counts['Category'], categories=order, ordered=True)
        cat_counts = cat_counts.sort_values('Category')
        
        # Create bar chart
        colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
        sns.barplot(x='Count', y='Category', data=cat_counts, palette=colors, ax=ax2)
        
        # Add data labels
        for i, v in enumerate(cat_counts['Count']):
            ax2.text(v + 1, i, str(v), va='center')
        
        # Format plot
        ax2.set_title('Player Performance vs. Expectations', fontsize=14)
        ax2.set_xlabel('Number of Players', fontsize=12)
        ax2.set_ylabel('')
    else:
        # Alternative plot if performance categories aren't available
        if 'ADP_vs_Performance' in df.columns:
            sns.histplot(df['ADP_vs_Performance'], kde=True, ax=ax2)
            ax2.axvline(0, color='red', linestyle='--')
            ax2.set_title('ADP Rank - Actual Rank Distribution', fontsize=14)
            ax2.set_xlabel('ADP Rank - Actual Rank (+ = Overperformed)', fontsize=12)
    
    plt.tight_layout()
    fig.suptitle('Pre-Season Expectations vs. Actual Performance', fontsize=16)
    plt.subplots_adjust(top=0.9)

def plot_consistency_analysis(df: pd.DataFrame, std_dev_col: str = None) -> None:
    """
    Plot player consistency analysis if weekly data is available.
    
    Args:
        df: Player performance dataframe
        std_dev_col: Name of column containing standard deviation of weekly points
    """
    # If no standard deviation column is specified, check for common names
    if std_dev_col is None:
        possible_cols = ['StdDev', 'Std_Dev', 'Weekly_StdDev', 'PPG_StdDev']
        for col in possible_cols:
            if col in df.columns:
                std_dev_col = col
                break
    
    # If we still don't have std dev data, we can't do this analysis
    if std_dev_col is None or std_dev_col not in df.columns:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "Weekly standard deviation data not available", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
        return
    
    # Create a figure with one subplot per position
    positions = ['QB', 'RB', 'WR', 'TE']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, pos in enumerate(positions):
        pos_df = df[df['FantPos'] == pos].copy()
        
        # Skip if no players for this position
        if pos_df.empty:
            axes[i].text(0.5, 0.5, f"No {pos} data available", 
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[i].transAxes, fontsize=14)
            continue
        
        # Create scatter plot of PPG vs std dev
        scatter = axes[i].scatter(pos_df['Half_PPR_PPG'], pos_df[std_dev_col], 
                                 alpha=0.7, s=60, c=pos_df['Half_PPR'], cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[i])
        cbar.set_label('Total Half-PPR Points')
        
        # Add player names for top performers
        top_players = pos_df.sort_values('Half_PPR', ascending=False).head(5)
        for _, row in top_players.iterrows():
            axes[i].annotate(row['Player'], 
                            (row['Half_PPR_PPG'], row[std_dev_col]),
                            xytext=(5, 5), textcoords='offset points')
        
        # Draw quadrant lines at median values
        median_ppg = pos_df['Half_PPR_PPG'].median()
        median_std = pos_df[std_dev_col].median()
        
        axes[i].axhline(median_std, color='gray', linestyle='--', alpha=0.5)
        axes[i].axvline(median_ppg, color='gray', linestyle='--', alpha=0.5)
        
        # Label quadrants
        axes[i].text(pos_df['Half_PPR_PPG'].max() * 0.7, pos_df[std_dev_col].min() * 1.5, 
                    "Consistent Stars", fontweight='bold', ha='center')
        axes[i].text(pos_df['Half_PPR_PPG'].min() * 1.5, pos_df[std_dev_col].min() * 1.5, 
                    "Consistent Low Scorers", fontweight='bold', ha='center')
        axes[i].text(pos_df['Half_PPR_PPG'].max() * 0.7, pos_df[std_dev_col].max() * 0.8, 
                    "Boom/Bust Stars", fontweight='bold', ha='center')
        axes[i].text(pos_df['Half_PPR_PPG'].min() * 1.5, pos_df[std_dev_col].max() * 0.8, 
                    "Boom/Bust Low Scorers", fontweight='bold', ha='center')
        
        # Format plot
        axes[i].set_title(f'{pos} Consistency Analysis', fontsize=14)
        axes[i].set_xlabel('Half-PPR Points Per Game', fontsize=12)
        axes[i].set_ylabel('Weekly Standard Deviation', fontsize=12)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.suptitle('Player Consistency Analysis by Position', fontsize=16)
