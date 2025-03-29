import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import matplotlib.ticker as mticker

def plot_vorp_by_position(df: pd.DataFrame) -> None:
    """
    Plot Value Over Replacement Player (VORP) by position.
    
    Args:
        df: Player dataframe with VORP metrics
    """
    if df is None or df.empty:
        plt.figure(figsize=(10, 6))
        plt.title("VORP Analysis Not Available")
        plt.text(0.5, 0.5, "No data available", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        return

    if 'VORP' not in df.columns or 'FantPos' not in df.columns:
        plt.figure(figsize=(10, 6))
        plt.title("VORP Analysis Not Available")
        plt.text(0.5, 0.5, "Missing VORP or position data", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        return
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Box plot of VORP by position
    position_order = ['QB', 'RB', 'WR', 'TE']
    palette = {'QB': '#FF9999', 'RB': '#99FF99', 'WR': '#9999FF', 'TE': '#FFFF99'}
    
    # Filter for players with positive VORP for better visualization
    plot_df = df[df['VORP'] > 0].copy()
    
    if plot_df.empty:
        plt.figure(figsize=(10, 6))
        plt.title("VORP Analysis Not Available")
        plt.text(0.5, 0.5, "No players with positive VORP", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        return
    
    sns.boxplot(x='FantPos', y='VORP', data=plot_df, 
               order=position_order, palette=palette, ax=ax1)
    
    # Add swarm plot for individual players
    sns.swarmplot(x='FantPos', y='VORP', data=plot_df, 
                 order=position_order, color='black', size=4, alpha=0.5, ax=ax1)
    
    # Format plot
    ax1.set_title('VORP Distribution by Position', fontsize=14)
    ax1.set_xlabel('Position', fontsize=12)
    ax1.set_ylabel('Value Over Replacement Player (VORP)', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Top 5 players by VORP for each position
    pos_colors = []
    positions_plotted = []
    
    for i, pos in enumerate(position_order):
        try:
            pos_df = df[df['FantPos'] == pos].sort_values('VORP', ascending=False).head(5)
            
            if pos_df.empty:
                continue
                
            positions_plotted.append(pos)
            
            # Get position color
            pos_color = palette.get(pos, f'C{i}')
            pos_colors.extend([pos_color] * len(pos_df))
            
            # Plot horizontal bars
            bars = ax2.barh(
                y=[f"{row['Player']} ({pos})" for _, row in pos_df.iterrows()],
                width=pos_df['VORP'],
                color=pos_color,
                alpha=0.7,
                label=pos
            )
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                        f"{width:.1f}", va='center')
        except Exception as e:
            print(f"Error plotting {pos}: {str(e)}")
            continue
    
    # Format plot
    ax2.set_title('Top 5 Players by VORP for Each Position', fontsize=14)
    ax2.set_xlabel('Value Over Replacement Player (VORP)', fontsize=12)
    ax2.set_ylabel('')
    ax2.grid(axis='x', alpha=0.3)
    if positions_plotted:
        ax2.legend(positions_plotted)
    
    plt.tight_layout()
    fig.suptitle('Positional Value Analysis', fontsize=16)
    plt.subplots_adjust(top=0.9)

def plot_positional_scarcity(scarcity_results: Dict[str, pd.DataFrame]) -> None:
    """
    Plot positional scarcity analysis.
    
    Args:
        scarcity_results: Dictionary of dataframes with positional scarcity analysis
    """
    if not scarcity_results:
        plt.figure(figsize=(10, 6))
        plt.title("Positional Scarcity Analysis Not Available")
        plt.text(0.5, 0.5, "No scarcity data available", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        return

    # Check if required data is available
    if not any(key.endswith('_rank_diffs') for key in scarcity_results.keys()):
        plt.figure(figsize=(10, 6))
        plt.title("Positional Scarcity Analysis Not Available")
        plt.text(0.5, 0.5, "Missing required scarcity data", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        return
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Positional drop-offs (rank differentials)
    positions = ['QB', 'RB', 'WR', 'TE']
    colors = ['#FF9999', '#99FF99', '#9999FF', '#FFFF99']
    
    for i, pos in enumerate(positions):
        try:
            key = f"{pos}_rank_diffs"
            if key not in scarcity_results:
                continue
                
            rank_diffs = scarcity_results[key]
            if rank_diffs.empty or 'Point_Diff' not in rank_diffs.columns:
                continue
                
            # Get positional drop-offs
            x = rank_diffs['Rank_Higher'].values
            y = rank_diffs['Point_Diff'].values
            
            # Plot
            ax1.plot(x, y, 'o-', label=pos, color=colors[i], alpha=0.7)
            
            # Add annotations for significant drop-offs (top 3)
            significant_drops = rank_diffs.sort_values('Point_Diff', ascending=False).head(3)
            for _, row in significant_drops.iterrows():
                if row['Point_Diff'] > 5:  # Only annotate significant drops
                    ax1.annotate(
                        f"{row['Rank_Higher']}-{row['Rank_Lower']}", 
                        (row['Rank_Higher'], row['Point_Diff']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8
                    )
        except Exception as e:
            print(f"Error plotting {pos} rank differentials: {str(e)}")
            continue
    
    # Format plot
    ax1.set_title('Point Differential Between Adjacent Ranks', fontsize=14)
    ax1.set_xlabel('Position Rank', fontsize=12)
    ax1.set_ylabel('Points Differential', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(title='Position')
    
    # Plot 2: Performance cliff visualization
    plt_colors = []
    
    # Check if position value adjustments are available
    if 'position_value_adjustments' in scarcity_results:
        try:
            value_adj = scarcity_results['position_value_adjustments']
            
            # Create bar chart of value adjustment factors
            if 'Value_Adjustment_Factor' in value_adj.columns and not value_adj.empty:
                # Sort by adjustment factor
                value_adj = value_adj.sort_values('Value_Adjustment_Factor', ascending=False)
                
                bars = ax2.bar(
                    value_adj['Position'],
                    value_adj['Value_Adjustment_Factor'],
                    color=colors[:len(value_adj)],
                    alpha=0.7
                )
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.05, 
                            f"{height:.2f}", ha='center', va='bottom')
                
                # Format plot
                ax2.set_title('Position Value Adjustment Factors', fontsize=14)
                ax2.set_xlabel('Position', fontsize=12)
                ax2.set_ylabel('Value Adjustment Factor', fontsize=12)
                ax2.grid(axis='y', alpha=0.3)
                
                # Add a horizontal line at 1.0 (neutral adjustment)
                ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.7)
            else:
                ax2.text(0.5, 0.5, "No value adjustment data available", 
                        ha='center', va='center', transform=ax2.transAxes)
        except Exception as e:
            print(f"Error plotting value adjustments: {str(e)}")
            ax2.text(0.5, 0.5, "Error plotting value adjustments", 
                    ha='center', va='center', transform=ax2.transAxes)
    else:
        # Alternative plot: performance distribution across positions
        for i, pos in enumerate(positions):
            try:
                key = f"{pos}_tiers"
                if key not in scarcity_results:
                    continue
                    
                pos_df = scarcity_results[key]
                if pos_df.empty or 'Half_PPR' not in pos_df.columns:
                    continue
                
                # Sort players by points
                pos_df = pos_df.sort_values('Half_PPR', ascending=False).reset_index(drop=True)
                pos_df['Rank'] = pos_df.index + 1
                
                # Plot
                ax2.plot(pos_df['Rank'], pos_df['Half_PPR'], 
                        'o-', label=pos, color=colors[i], alpha=0.7)
            except Exception as e:
                print(f"Error plotting {pos} tiers: {str(e)}")
                continue
        
        # Format plot
        ax2.set_title('Performance Drop-off by Position Rank', fontsize=14)
        ax2.set_xlabel('Position Rank', fontsize=12)
        ax2.set_ylabel('Half-PPR Points', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(title='Position')
    
    plt.tight_layout()
    fig.suptitle('Positional Scarcity Analysis', fontsize=16)
    plt.subplots_adjust(top=0.9)

def plot_position_group_comparison(group_results: Dict[str, pd.DataFrame]) -> None:
    """
    Plot comparison between position groups (RB/WR vs. QB/TE).
    
    Args:
        group_results: Dictionary of dataframes with position group analysis
    """
    # Check if required data is available
    req_keys = ['Primary_Skill_position_stats', 'Onesies_position_stats']
    if not group_results or not all(key in group_results for key in req_keys):
        plt.figure(figsize=(10, 6))
        plt.title("Position Group Comparison Not Available")
        plt.text(0.5, 0.5, "Missing required position group data", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        return
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Average points by position
    primary = group_results['Primary_Skill_position_stats']
    onesies = group_results['Onesies_position_stats']
    
    # Combine data
    all_pos = pd.concat([primary, onesies])
    
    # Check for required columns
    if 'Avg_Points' not in all_pos.columns or 'Position' not in all_pos.columns:
        ax1.text(0.5, 0.5, "Missing required columns for position comparison", 
                 ha='center', va='center', transform=ax1.transAxes)
    else:
        # Sort by average points
        all_pos = all_pos.sort_values('Avg_Points', ascending=False)
        
        # Create bar colors
        colors = []
        for pos in all_pos['Position']:
            if pos in ['RB', 'WR']:
                colors.append('#99FF99' if pos == 'RB' else '#9999FF')
            else:
                colors.append('#FF9999' if pos == 'QB' else '#FFFF99')
        
        # Create bar chart
        bars = ax1.bar(
            all_pos['Position'],
            all_pos['Avg_Points'],
            color=colors,
            alpha=0.7
        )
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 5, 
                    f"{height:.1f}", ha='center', va='bottom')
        
        # Format plot
        ax1.set_title('Average Points by Position', fontsize=14)
        ax1.set_xlabel('Position', fontsize=12)
        ax1.set_ylabel('Average Half-PPR Points', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Draft capital allocation
    if 'draft_capital_allocation' in group_results:
        alloc = group_results['draft_capital_allocation']
        
        if 'Recommended_Draft_Capital' in alloc.columns and 'Group' in alloc.columns:
            # Create pie chart
            labels = [f"{row['Group']} ({row['Positions']})" for _, row in alloc.iterrows()]
            sizes = alloc['Recommended_Draft_Capital']
            colors = ['#7986CB', '#4CAF50']  # Blue for Primary, Green for Onesies
            
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                   startangle=90, shadow=False, wedgeprops={'alpha': 0.7})
            
            # Format plot
            ax2.set_title('Recommended Draft Capital Allocation', fontsize=14)
            ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    else:
        # Alternative plot: Top tier point differences
        if 'Top_Tier_Avg' in all_pos.columns:
            # Sort by top tier average
            all_pos = all_pos.sort_values('Top_Tier_Avg', ascending=False)
            
            # Create bar colors
            colors = []
            for pos in all_pos['Position']:
                if pos in ['RB', 'WR']:
                    colors.append('#99FF99' if pos == 'RB' else '#9999FF')
                else:
                    colors.append('#FF9999' if pos == 'QB' else '#FFFF99')
            
            # Create bar chart
            bars = ax2.bar(
                all_pos['Position'],
                all_pos['Top_Tier_Avg'],
                color=colors,
                alpha=0.7
            )
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 5, 
                        f"{height:.1f}", ha='center', va='bottom')
            
            # Format plot
            ax2.set_title('Top Tier Average Points by Position', fontsize=14)
            ax2.set_xlabel('Position', fontsize=12)
            ax2.set_ylabel('Average Half-PPR Points (Top Tier)', fontsize=12)
            ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.suptitle('Position Group Comparison', fontsize=16)
    plt.subplots_adjust(top=0.9)

def plot_positional_advantage(df: pd.DataFrame) -> None:
    """
    Plot positional advantage based on VORP or similar metrics.
    
    Args:
        df: Player dataframe with positional value metrics
    """
    if 'VORP' not in df.columns or 'FantPos' not in df.columns:
        plt.figure(figsize=(10, 6))
        plt.title("Positional Advantage Analysis Not Available")
        plt.text(0.5, 0.5, "Missing VORP or position data", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        return
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: VORP Cumulative by Position
    positions = ['QB', 'RB', 'WR', 'TE']
    colors = ['#FF9999', '#99FF99', '#9999FF', '#FFFF99']
    
    for i, pos in enumerate(positions):
        pos_df = df[df['FantPos'] == pos].copy()
        
        if pos_df.empty:
            continue
            
        # Sort by VORP
        pos_df = pos_df.sort_values('VORP', ascending=False).reset_index(drop=True)
        pos_df['Rank'] = pos_df.index + 1
        
        # Calculate cumulative VORP
        pos_df['Cumulative_VORP'] = pos_df['VORP'].cumsum()
        
        # Plot
        ax1.plot(pos_df['Rank'], pos_df['Cumulative_VORP'], 
                label=pos, color=colors[i], linewidth=2, alpha=0.7)
        
        # Add annotations for key points (starter cutoffs)
        starter_counts = {'QB': 12, 'RB': 24, 'WR': 24, 'TE': 12}
        if pos in starter_counts and len(pos_df) >= starter_counts[pos]:
            cutoff_rank = starter_counts[pos]
            cutoff_vorp = pos_df.loc[pos_df['Rank'] == cutoff_rank, 'Cumulative_VORP'].values
            if len(cutoff_vorp) > 0:
                ax1.plot(cutoff_rank, cutoff_vorp[0], 'o', color=colors[i], markersize=8)
                ax1.annotate(
                    f"{pos}{cutoff_rank}",
                    (cutoff_rank, cutoff_vorp[0]),
                    xytext=(5, 5), textcoords='offset points'
                )
    
    # Format plot
    ax1.set_title('Cumulative VORP by Position', fontsize=14)
    ax1.set_xlabel('Position Rank', fontsize=12)
    ax1.set_ylabel('Cumulative VORP', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(title='Position')
    
    # Plot 2: Relative Value Index
    # Calculate position adjustments internally for this plot
    pos_adj = {}
    pos_data = []
    
    for i, pos in enumerate(positions):
        pos_df = df[df['FantPos'] == pos].copy()
        
        if pos_df.empty:
            continue
        
        # Calculate value metrics for startable players
        starter_counts = {'QB': 12, 'RB': 24, 'WR': 24, 'TE': 12}
        if pos in starter_counts:
            # Get startable players
            starters = min(starter_counts[pos], len(pos_df))
            startable_df = pos_df.sort_values('Half_PPR', ascending=False).head(starters)
            
            # Calculate average and std dev
            avg_points = startable_df['Half_PPR'].mean()
            std_dev = startable_df['Half_PPR'].std()
            
            pos_adj[pos] = std_dev
            pos_data.append({
                'Position': pos,
                'Avg_Points': avg_points,
                'Std_Dev': std_dev,
                'Starter_Count': starters
            })
    
    if pos_data:
        # Create dataframe
        pos_df = pd.DataFrame(pos_data)
        
        # Normalize standard deviations for value index
        total_std = pos_df['Std_Dev'].sum()
        pos_df['Value_Index'] = pos_df['Std_Dev'] / total_std * len(pos_df)
        
        # Create bar chart of value index
        bars = ax2.bar(
            pos_df['Position'],
            pos_df['Value_Index'],
            color=colors[:len(pos_df)],
            alpha=0.7
        )
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.05, 
                    f"{height:.2f}", ha='center', va='bottom')
        
        # Add starter count as text
        for i, row in pos_df.iterrows():
            ax2.text(i, 0.1, f"Top {row['Starter_Count']}", ha='center')
        
        # Format plot
        ax2.set_title('Positional Value Index', fontsize=14)
        ax2.set_xlabel('Position', fontsize=12)
        ax2.set_ylabel('Value Index (Higher = More Valuable)', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add a horizontal line at 1.0 (average value)
        ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig.suptitle('Positional Advantage Analysis', fontsize=16)
    plt.subplots_adjust(top=0.9)
