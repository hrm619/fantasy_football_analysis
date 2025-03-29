import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict

def plot_player_tiers(df: pd.DataFrame, position: str) -> None:
    """
    Plot player tiers for a specific position.
    
    Args:
        df: Player dataframe with tier assignments
        position: Position to plot
    """
    if df is None or df.empty:
        plt.figure(figsize=(10, 6))
        plt.title(f"{position} Tiers Not Available")
        plt.text(0.5, 0.5, "No data available", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        return

    if 'Tier' not in df.columns or 'Half_PPR' not in df.columns:
        plt.figure(figsize=(10, 6))
        plt.title(f"{position} Tiers Not Available")
        plt.text(0.5, 0.5, "Missing required columns", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        return
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Box plot of points by tier
    # Sort by tier number
    df['Tier_Num'] = df['Tier'].str.extract(r'Tier (\d+)').astype(int)
    df = df.sort_values('Tier_Num')
    
    # Get unique tiers and create color palette
    tiers = sorted(df['Tier'].unique())
    colors = sns.color_palette("husl", len(tiers))
    
    # Create box plot
    sns.boxplot(x='Tier', y='Half_PPR', data=df, palette=colors, ax=ax1)
    
    # Add swarm plot for individual players
    sns.swarmplot(x='Tier', y='Half_PPR', data=df, color='black', size=4, alpha=0.5, ax=ax1)
    
    # Format plot
    ax1.set_title(f'{position} Performance by Tier', fontsize=14)
    ax1.set_xlabel('Tier', fontsize=12)
    ax1.set_ylabel('Half-PPR Points', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Top 5 players in each tier
    for i, tier in enumerate(tiers):
        tier_df = df[df['Tier'] == tier].sort_values('Half_PPR', ascending=False).head(5)
        
        # Create horizontal bars
        bars = ax2.barh(
            y=[f"{row['Player']} ({row['Team_std']})" for _, row in tier_df.iterrows()],
            width=tier_df['Half_PPR'],
            color=colors[i],
            alpha=0.7,
            label=tier
        )
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f"{width:.1f}", va='center')
    
    # Format plot
    ax2.set_title('Top 5 Players in Each Tier', fontsize=14)
    ax2.set_xlabel('Half-PPR Points', fontsize=12)
    ax2.set_ylabel('')
    ax2.grid(axis='x', alpha=0.3)
    ax2.legend(title='Tier')
    
    plt.tight_layout()
    fig.suptitle(f'{position} Tier Analysis', fontsize=16)
    plt.subplots_adjust(top=0.9)

def plot_player_archetypes(df: pd.DataFrame, position: str, features: List[str]) -> None:
    """
    Plot player archetypes for a specific position.
    
    Args:
        df: Player dataframe with archetype assignments
        position: Position to plot
        features: List of features used for archetype identification
    """
    if df is None or df.empty:
        plt.figure(figsize=(10, 6))
        plt.title(f"{position} Archetypes Not Available")
        plt.text(0.5, 0.5, "No data available", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        return

    if 'Archetype' not in df.columns or 'Half_PPR' not in df.columns:
        plt.figure(figsize=(10, 6))
        plt.title(f"{position} Archetypes Not Available")
        plt.text(0.5, 0.5, "Missing required columns", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        return
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Box plot of points by archetype
    # Get unique archetypes and create color palette
    archetypes = sorted(df['Archetype'].unique())
    colors = sns.color_palette("husl", len(archetypes))
    
    # Create box plot
    sns.boxplot(x='Archetype', y='Half_PPR', data=df, palette=colors, ax=ax1)
    
    # Add swarm plot for individual players
    sns.swarmplot(x='Archetype', y='Half_PPR', data=df, color='black', size=4, alpha=0.5, ax=ax1)
    
    # Format plot
    ax1.set_title(f'{position} Performance by Archetype', fontsize=14)
    ax1.set_xlabel('Archetype', fontsize=12)
    ax1.set_ylabel('Half-PPR Points', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Feature importance for each archetype
    if len(features) > 0:
        # Calculate feature means for each archetype
        feature_means = df.groupby('Archetype')[features].mean()
        
        # Normalize features for better visualization
        feature_means_norm = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min())
        
        # Create heatmap
        sns.heatmap(feature_means_norm, annot=True, cmap='RdYlBu_r', center=0.5, ax=ax2)
        
        # Format plot
        ax2.set_title('Archetype Feature Importance', fontsize=14)
        ax2.set_xlabel('Features', fontsize=12)
        ax2.set_ylabel('Archetype', fontsize=12)
    else:
        # Alternative plot: Top 5 players in each archetype
        for i, archetype in enumerate(archetypes):
            arch_df = df[df['Archetype'] == archetype].sort_values('Half_PPR', ascending=False).head(5)
            
            # Create horizontal bars
            bars = ax2.barh(
                y=[f"{row['Player']} ({row['Team_std']})" for _, row in arch_df.iterrows()],
                width=arch_df['Half_PPR'],
                color=colors[i],
                alpha=0.7,
                label=archetype
            )
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                        f"{width:.1f}", va='center')
        
        # Format plot
        ax2.set_title('Top 5 Players in Each Archetype', fontsize=14)
        ax2.set_xlabel('Half-PPR Points', fontsize=12)
        ax2.set_ylabel('')
        ax2.grid(axis='x', alpha=0.3)
        ax2.legend(title='Archetype')
    
    plt.tight_layout()
    fig.suptitle(f'{position} Archetype Analysis', fontsize=16)
    plt.subplots_adjust(top=0.9)

def plot_tier_expectations(df: pd.DataFrame) -> None:
    """
    Plot tier performance vs. expectations.
    
    Args:
        df: Dataframe with tier expectations analysis
    """
    if df is None or df.empty:
        plt.figure(figsize=(10, 6))
        plt.title("Tier Expectations Not Available")
        plt.text(0.5, 0.5, "No data available", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        return

    if 'Position' not in df.columns or 'Tier' not in df.columns:
        plt.figure(figsize=(10, 6))
        plt.title("Tier Expectations Not Available")
        plt.text(0.5, 0.5, "Missing required columns", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        return
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Points per ADP by position and tier
    positions = df['Position'].unique()
    colors = sns.color_palette("husl", len(positions))
    
    for i, pos in enumerate(positions):
        pos_df = df[df['Position'] == pos]
        
        # Sort by tier number
        pos_df['Tier_Num'] = pos_df['Tier'].str.extract(r'Tier (\d+)').astype(int)
        pos_df = pos_df.sort_values('Tier_Num')
        
        # Plot
        ax1.plot(pos_df['Tier'], pos_df['Points_per_ADP'], 
                'o-', label=pos, color=colors[i], alpha=0.7)
        
        # Add value labels
        for _, row in pos_df.iterrows():
            ax1.text(row['Tier'], row['Points_per_ADP'], 
                    f"{row['Points_per_ADP']:.2f}", 
                    ha='center', va='bottom')
    
    # Format plot
    ax1.set_title('Points per ADP by Position and Tier', fontsize=14)
    ax1.set_xlabel('Tier', fontsize=12)
    ax1.set_ylabel('Points per ADP', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(title='Position')
    
    # Plot 2: Success rate by position and tier
    if 'Success_Rate' in df.columns:
        for i, pos in enumerate(positions):
            pos_df = df[df['Position'] == pos]
            
            # Sort by tier number
            pos_df['Tier_Num'] = pos_df['Tier'].str.extract(r'Tier (\d+)').astype(int)
            pos_df = pos_df.sort_values('Tier_Num')
            
            # Plot
            ax2.plot(pos_df['Tier'], pos_df['Success_Rate'] * 100, 
                    'o-', label=pos, color=colors[i], alpha=0.7)
            
            # Add value labels
            for _, row in pos_df.iterrows():
                ax2.text(row['Tier'], row['Success_Rate'] * 100, 
                        f"{row['Success_Rate']*100:.1f}%", 
                        ha='center', va='bottom')
        
        # Format plot
        ax2.set_title('Success Rate by Position and Tier', fontsize=14)
        ax2.set_xlabel('Tier', fontsize=12)
        ax2.set_ylabel('Success Rate (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(title='Position')
    else:
        # Alternative plot: Expected vs Actual Points
        for i, pos in enumerate(positions):
            pos_df = df[df['Position'] == pos]
            
            # Sort by tier number
            pos_df['Tier_Num'] = pos_df['Tier'].str.extract(r'Tier (\d+)').astype(int)
            pos_df = pos_df.sort_values('Tier_Num')
            
            # Plot
            ax2.plot(pos_df['Tier'], pos_df['Avg_Points'], 
                    'o-', label=f"{pos} Actual", color=colors[i], alpha=0.7)
            
            if 'Expected_Points' in pos_df.columns:
                ax2.plot(pos_df['Tier'], pos_df['Expected_Points'], 
                        'o--', label=f"{pos} Expected", color=colors[i], alpha=0.5)
            
            # Add value labels
            for _, row in pos_df.iterrows():
                ax2.text(row['Tier'], row['Avg_Points'], 
                        f"{row['Avg_Points']:.1f}", 
                        ha='center', va='bottom')
        
        # Format plot
        ax2.set_title('Expected vs Actual Points by Position and Tier', fontsize=14)
        ax2.set_xlabel('Tier', fontsize=12)
        ax2.set_ylabel('Points', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(title='Position')
    
    plt.tight_layout()
    fig.suptitle('Tier Performance vs Expectations', fontsize=16)
    plt.subplots_adjust(top=0.9)

# TODO: Implement other tiering visualization functions
