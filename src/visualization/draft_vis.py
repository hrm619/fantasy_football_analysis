import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional

def plot_round_performance(df: pd.DataFrame) -> None:
    """
    Plot fantasy performance by draft round.
    
    Args:
        df: Round performance statistics dataframe
    """
    if df.empty or 'Draft_Round' not in df.columns:
        plt.figure(figsize=(10, 6))
        plt.title("Round Performance Analysis Not Available")
        plt.text(0.5, 0.5, "Missing draft round performance data", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        return
    
    # Clear any existing figures
    plt.clf()
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot 1: Average points by round
    if 'Avg_Points' in df.columns:
        # Create bar chart
        bars = ax1.bar(
            df['Draft_Round'],
            df['Avg_Points'],
            color='steelblue',
            alpha=0.7
        )
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 2, 
                    f"{height:.1f}", ha='center', va='bottom')
        
        # Add trendline
        x = df['Draft_Round']
        y = df['Avg_Points']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax1.plot(x, p(x), "r--", alpha=0.7)
        
        # Format plot
        ax1.set_title('Average Fantasy Points by Draft Round', fontsize=14)
        ax1.set_xlabel('Draft Round', fontsize=12)
        ax1.set_ylabel('Average Half-PPR Points', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # Set x-axis to integer ticks only
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    else:
        ax1.text(0.5, 0.5, "Missing average points data", 
                 ha='center', va='center', transform=ax1.transAxes)
    
    # Plot 2: Success and bust rates by round
    if 'Success_Rate' in df.columns and 'Bust_Rate' in df.columns:
        # Create line chart with two y-axes
        color1, color2 = 'tab:blue', 'tab:red'
        
        # Success rate
        line1 = ax2.plot(df['Draft_Round'], df['Success_Rate'], 'o-', 
                        color=color1, label='Success Rate', linewidth=2)
        ax2.set_xlabel('Draft Round', fontsize=12)
        ax2.set_ylabel('Success Rate (%)', color=color1, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color1)
        
        # Create second y-axis
        ax2_2 = ax2.twinx()
        
        # Bust rate
        line2 = ax2_2.plot(df['Draft_Round'], df['Bust_Rate'], 'o-', 
                          color=color2, label='Bust Rate', linewidth=2)
        ax2_2.set_ylabel('Bust Rate (%)', color=color2, fontsize=12)
        ax2_2.tick_params(axis='y', labelcolor=color2)
        
        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper center')
        
        # Format plot
        ax2.set_title('Success and Bust Rates by Draft Round', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Set x-axis to integer ticks only
        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    else:
        ax2.text(0.5, 0.5, "Missing success/bust rate data", 
                 ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    fig.suptitle('Draft Round Analysis', fontsize=16)
    plt.subplots_adjust(top=0.9)

def plot_position_value_by_round(df: pd.DataFrame) -> None:
    """
    Plot position value analysis by draft round.
    
    Args:
        df: Position round statistics dataframe
    """
    if df.empty or 'Draft_Round' not in df.columns or 'Position' not in df.columns:
        plt.figure(figsize=(10, 6))
        plt.title("Position Value by Round Analysis Not Available")
        plt.text(0.5, 0.5, "Missing position round data", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        return
    
    # Clear any existing figures
    plt.clf()
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot 1: Position performance by round heatmap
    if 'Avg_Points' in df.columns:
        # Pivot the data for heatmap
        pivot_df = df.pivot_table(
            index='Draft_Round', 
            columns='Position', 
            values='Avg_Points'
        )
        
        # Create heatmap
        sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='viridis', 
                   linewidths=.5, ax=ax1, cbar_kws={'label': 'Avg Half-PPR Points'})
        
        # Format plot
        ax1.set_title('Average Points by Position and Round', fontsize=14)
        ax1.set_xlabel('Position', fontsize=12)
        ax1.set_ylabel('Draft Round', fontsize=12)
    else:
        ax1.text(0.5, 0.5, "Missing average points data", 
                 ha='center', va='center', transform=ax1.transAxes)
    
    # Plot 2: Success rate by position and round
    if 'Success_Rate' in df.columns:
        # Pivot the data for heatmap
        pivot_df = df.pivot_table(
            index='Draft_Round', 
            columns='Position', 
            values='Success_Rate'
        )
        
        # Create heatmap
        sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdYlGn', 
                   linewidths=.5, ax=ax2, cbar_kws={'label': 'Success Rate (%)'})
        
        # Format plot
        ax2.set_title('Success Rate by Position and Round', fontsize=14)
        ax2.set_xlabel('Position', fontsize=12)
        ax2.set_ylabel('Draft Round', fontsize=12)
    else:
        ax2.text(0.5, 0.5, "Missing success rate data", 
                 ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    fig.suptitle('Position Value by Draft Round', fontsize=16)
    plt.subplots_adjust(top=0.9)

def plot_vbd_rankings(df: pd.DataFrame) -> None:
    """
    Plot VBD rankings and distribution by position.
    
    Args:
        df: Player dataframe with VBD rankings
    """
    if 'VBD' not in df.columns or 'FantPos' not in df.columns:
        plt.figure(figsize=(10, 6))
        plt.title("VBD Rankings Not Available")
        plt.text(0.5, 0.5, "Missing VBD or position data", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        return
    
    # Clear any existing figures
    plt.clf()
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot 1: Top 20 players by VBD
    top_20 = df.nlargest(20, 'VBD')
    
    # Define position colors
    pos_colors = {'QB': '#FF9999', 'RB': '#99FF99', 'WR': '#9999FF', 'TE': '#FFFF99'}
    
    # Create horizontal bar chart
    bars = ax1.barh(
        range(len(top_20)),
        top_20['VBD'],
        color=[pos_colors.get(pos, 'gray') for pos in top_20['FantPos']],
        alpha=0.7
    )
    
    # Add player labels
    for i, player in top_20.iterrows():
        ax1.text(1, top_20.index.get_loc(i), 
                f"{player['Player']} ({player['FantPos']})", 
                va='center')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f"{width:.1f}", va='center')
    
    # Format plot
    ax1.set_title('Top 20 Players by VBD', fontsize=14)
    ax1.set_xlabel('Value Based Drafting (VBD) Score', fontsize=12)
    ax1.set_ylabel('')
    ax1.invert_yaxis()  # Highest VBD at the top
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: VBD distribution by position
    if len(df) >= 10:  # Only show if we have enough data
        # Filter for players with positive VBD for better visualization
        plot_df = df[df['VBD'] > 0].copy()
        
        # Ensure we have data for all positions
        positions = ['QB', 'RB', 'WR', 'TE']
        for pos in positions:
            if pos not in plot_df['FantPos'].unique():
                # Add a dummy row with 0 VBD for missing positions
                plot_df = pd.concat([plot_df, pd.DataFrame({
                    'FantPos': [pos],
                    'VBD': [0]
                })], ignore_index=True)
        
        # Create violin plot
        sns.violinplot(x='FantPos', y='VBD', data=plot_df,
                      palette=pos_colors, inner='quartile', ax=ax2)
        
        # Add swarm plot only for positions with data
        for pos in positions:
            pos_data = plot_df[plot_df['FantPos'] == pos]
            if len(pos_data) > 0 and pos_data['VBD'].sum() > 0:
                sns.swarmplot(x='FantPos', y='VBD', data=pos_data,
                            color='black', size=3, alpha=0.5, ax=ax2)
        
        # Format plot
        ax2.set_title('VBD Distribution by Position', fontsize=14)
        ax2.set_xlabel('Position', fontsize=12)
        ax2.set_ylabel('Value Based Drafting (VBD) Score', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Insufficient data for distribution plot", 
                 ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    fig.suptitle('Value Based Drafting (VBD) Analysis', fontsize=16)
    plt.subplots_adjust(top=0.9)

def plot_adp_value_analysis(df: pd.DataFrame) -> None:
    """
    Plot analysis of ADP vs actual value to identify draft bargains and busts.
    
    Args:
        df: Player dataframe with ADP and performance metrics
    """
    if 'ADP' not in df.columns or 'Half_PPR' not in df.columns or 'FantPos' not in df.columns:
        plt.figure(figsize=(10, 6))
        plt.title("ADP Value Analysis Not Available")
        plt.text(0.5, 0.5, "Missing ADP or performance data", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        return
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: ADP vs. Performance scatter with bargain/bust zones
    positions = df['FantPos'].unique()
    colors = {'QB': '#FF9999', 'RB': '#99FF99', 'WR': '#9999FF', 'TE': '#FFFF99'}
    
    # Create scatter plot
    for pos in positions:
        pos_df = df[df['FantPos'] == pos]
        ax1.scatter(pos_df['ADP'], pos_df['Half_PPR'], 
                   label=pos, color=colors.get(pos, 'gray'), alpha=0.7, s=60)
    
    # Add diagonal line (y = x scaled to match ranges)
    max_adp = df['ADP'].max()
    max_pts = df['Half_PPR'].max()
    scale_factor = max_pts / max_adp
    
    x = np.linspace(0, max_adp, 100)
    y = x * scale_factor
    ax1.plot(x, y, 'k--', alpha=0.5, label='Expected Value')
    
    # Add bargain and bust zones
    bargain_y = x * scale_factor * 1.5
    bust_y = x * scale_factor * 0.5
    
    ax1.fill_between(x, bargain_y, max_pts + 10, alpha=0.1, color='green', label='Bargain Zone')
    ax1.fill_between(x, 0, bust_y, alpha=0.1, color='red', label='Bust Zone')
    
    # Annotate noteworthy outliers
    top_bargains = df[df['Half_PPR'] > df['ADP'] * scale_factor * 1.8].head(3)
    top_busts = df[df['Half_PPR'] < df['ADP'] * scale_factor * 0.4].head(3)
    
    for _, player in top_bargains.iterrows():
        ax1.annotate(
            player['Player'],
            (player['ADP'], player['Half_PPR']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8
        )
    
    for _, player in top_busts.iterrows():
        ax1.annotate(
            player['Player'],
            (player['ADP'], player['Half_PPR']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8
        )
    
    # Format plot
    ax1.set_title('ADP vs. Performance with Value Zones', fontsize=14)
    ax1.set_xlabel('Average Draft Position (ADP)', fontsize=12)
    ax1.set_ylabel('Half-PPR Points', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Value per ADP point by position
    # Calculate points per ADP point
    df['Points_per_ADP'] = df['Half_PPR'] / df['ADP']
    
    # Group by position
    pos_value = df.groupby('FantPos')['Points_per_ADP'].mean().reset_index()
    pos_value = pos_value.sort_values('Points_per_ADP', ascending=False)
    
    # Create bar chart
    bars = ax2.bar(
        pos_value['FantPos'],
        pos_value['Points_per_ADP'],
        color=[colors.get(pos, 'gray') for pos in pos_value['FantPos']],
        alpha=0.7
    )
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.1, 
                f"{height:.2f}", ha='center', va='bottom')
    
    # Format plot
    ax2.set_title('Fantasy Points per ADP Point by Position', fontsize=14)
    ax2.set_xlabel('Position', fontsize=12)
    ax2.set_ylabel('Points per ADP Point', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.suptitle('ADP Value Analysis', fontsize=16)
    plt.subplots_adjust(top=0.9)

def plot_draft_pick_analysis(df: pd.DataFrame, team_count: int = 12) -> None:
    """
    Analyze optimal draft picks by slot and round.
    
    Args:
        df: Player performance dataframe with ADP data
        team_count: Number of teams in the league
    """
    if 'ADP' not in df.columns or 'Half_PPR' not in df.columns or 'FantPos' not in df.columns:
        plt.figure(figsize=(10, 6))
        plt.title("Draft Pick Analysis Not Available")
        plt.text(0.5, 0.5, "Missing ADP or performance data", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        return
    
    # Add pick information
    analysis_df = df.copy()
    analysis_df['Draft_Round'] = np.ceil(analysis_df['ADP'] / team_count).astype(int)
    analysis_df['Pick_In_Round'] = ((analysis_df['ADP'] - 1) % team_count) + 1
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Performance by pick slot
    if 'Pick_In_Round' in analysis_df.columns:
        # Calculate average points by pick slot across all rounds
        pick_perf = analysis_df.groupby('Pick_In_Round')['Half_PPR'].mean().reset_index()
        
        # Create bar chart
        bars = ax1.bar(
            pick_perf['Pick_In_Round'],
            pick_perf['Half_PPR'],
            color='steelblue',
            alpha=0.7
        )
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 2, 
                    f"{height:.1f}", ha='center', va='bottom')
        
        # Format plot
        ax1.set_title('Average Performance by Draft Slot', fontsize=14)
        ax1.set_xlabel('Pick Position in Round', fontsize=12)
        ax1.set_ylabel('Average Half-PPR Points', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # Set x-axis to integer ticks
        ax1.set_xticks(range(1, team_count + 1))
    else:
        ax1.text(0.5, 0.5, "Unable to calculate pick positions", 
                 ha='center', va='center', transform=ax1.transAxes)
    
    # Plot 2: Position distribution by draft slot
    if 'Pick_In_Round' in analysis_df.columns:
        # Get position distribution by pick slot
        pos_counts = analysis_df.groupby(['Pick_In_Round', 'FantPos']).size().unstack().fillna(0)
        
        # Convert to percentages
        pos_pct = pos_counts.div(pos_counts.sum(axis=1), axis=0) * 100
        
        # Create stacked bar chart
        pos_pct.plot(kind='bar', stacked=True, ax=ax2, 
                    color=['#FF9999', '#99FF99', '#9999FF', '#FFFF99'])
        
        # Format plot
        ax2.set_title('Position Distribution by Draft Slot', fontsize=14)
        ax2.set_xlabel('Pick Position in Round', fontsize=12)
        ax2.set_ylabel('Percentage of Picks (%)', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend(title='Position')
    else:
        ax2.text(0.5, 0.5, "Unable to calculate position distribution", 
                 ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    fig.suptitle('Draft Pick Position Analysis', fontsize=16)
    plt.subplots_adjust(top=0.9)
