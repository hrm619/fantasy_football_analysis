"""
Dashboard callbacks module
Implements the callback functions for the Fantasy Football Analysis Dashboard
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path to import project modules
parent_dir = str(Path(__file__).resolve().parent.parent)
import sys
sys.path.append(parent_dir)

# Import project modules
from src.data.data_loader import load_config
from layouts import (
    create_overview_layout,
    create_performance_layout,
    create_position_layout,
    create_draft_layout,
    create_archetypes_layout,
    create_team_layout
)

# Load configuration
config = load_config()

# Load processed data paths
processed_data_path = config['data_paths']['processed_data']
draft_strategy_path = os.path.join(processed_data_path, 'draft_strategy')

def register_callbacks(app):
    """
    Register all callbacks for the dashboard
    
    Args:
        app: Dash application instance
    """
    
    # Tab content callback
    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "active_tab"),
    )
    def render_tab_content(active_tab):
        """
        Render content based on active tab
        
        Args:
            active_tab: Active tab ID
            
        Returns:
            dash.html component: Tab content
        """
        if active_tab == "tab-overview":
            return create_overview_layout()
        elif active_tab == "tab-performance":
            return create_performance_layout()
        elif active_tab == "tab-position":
            return create_position_layout()
        elif active_tab == "tab-draft":
            return create_draft_layout()
        elif active_tab == "tab-archetypes":
            return create_archetypes_layout()
        elif active_tab == "tab-team":
            return create_team_layout()
        
        return create_overview_layout()
    
    # Overview tab callbacks
    @app.callback(
        Output("key-insights", "children"),
        Input("tabs", "active_tab"),
    )
    def update_key_insights(active_tab):
        """Generate key insights for the overview tab"""
        if active_tab != "tab-overview":
            return html.Div()
        
        return html.Div([
            html.H5("Key Fantasy Insights for 2025"),
            html.Ul([
                html.Li("Position scarcity affects draft strategy significantly - elite RBs and WRs provide the most value"),
                html.Li("QB and TE positions show pronounced tiers, with significant drop-offs after the elite options"),
                html.Li("Draft position matters less than recognizing value within each round"),
                html.Li("Team context heavily influences individual player success - target players in favorable situations"),
                html.Li("Efficiency metrics help identify potential breakout candidates and regression risks"),
            ]),
        ])
    
    @app.callback(
        Output("draft-summary", "children"),
        Input("tabs", "active_tab"),
    )
    def update_draft_summary(active_tab):
        """Generate draft summary for the overview tab"""
        if active_tab != "tab-overview":
            return html.Div()
        
        return html.Div([
            html.H5("Draft Strategy Summary"),
            html.Ul([
                html.Li("Early rounds (1-3): Focus on elite RB/WR talent regardless of position"),
                html.Li("Middle rounds (4-8): Target high-upside players and positional value"),
                html.Li("Late rounds (9+): Seek out high-efficiency players in favorable team situations"),
                html.Li("Consider position tiers when making selections to maximize value"),
                html.Li("Draft capital allocation: ~70% to RB/WR positions, ~30% to QB/TE"),
            ]),
        ])
    
    @app.callback(
        Output("position-value-overview", "figure"),
        Input("tabs", "active_tab"),
    )
    def update_position_value_overview(active_tab):
        """Generate position value overview chart"""
        if active_tab != "tab-overview":
            return go.Figure()
        
        try:
            # Load VBD data
            vbd_df = pd.read_csv(os.path.join(processed_data_path, 'vbd_rankings.csv'))
            
            # Create position summary
            pos_summary = vbd_df.groupby('FantPos').agg({
                'VBD': ['mean', 'sum', 'max', 'count'],
                'Half_PPR': ['mean', 'max']
            }).reset_index()
            
            # Flatten multi-index columns
            pos_summary.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in pos_summary.columns]
            
            # Create subplot
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "bar"}, {"type": "bar"}]],
                subplot_titles=("Average VBD by Position", "Top Player VBD by Position")
            )
            
            # Add average VBD by position
            fig.add_trace(
                go.Bar(
                    x=pos_summary['FantPos'],
                    y=pos_summary['VBD_mean'],
                    name="Avg VBD",
                    marker_color=['#FF9999', '#99FF99', '#9999FF', '#FFFF99'],
                ),
                row=1, col=1
            )
            
            # Add max VBD by position
            fig.add_trace(
                go.Bar(
                    x=pos_summary['FantPos'],
                    y=pos_summary['VBD_max'],
                    name="Max VBD",
                    marker_color=['#FF9999', '#99FF99', '#9999FF', '#FFFF99'],
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=500,
                showlegend=False,
                title_text="Position Value Comparison (2024 Season)",
                title_x=0.5,
            )
            
            return fig
        except Exception as e:
            # Return empty figure with error message
            fig = go.Figure()
            fig.update_layout(
                title="Error loading position value data",
                annotations=[dict(text=str(e), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)]
            )
            return fig
    
    # Performance tab callbacks
    @app.callback(
        Output("performance-chart", "figure"),
        [Input("position-filter", "value"),
         Input("performance-metric", "value")]
    )
    def update_performance_chart(position, metric):
        """Generate player performance chart based on filters"""
        try:
            # Load performance data
            performance_df = pd.read_csv(os.path.join(processed_data_path, 'player_performance.csv'))
            
            # Filter by position if needed
            if position != "ALL":
                filtered_df = performance_df[performance_df['FantPos'] == position]
            else:
                filtered_df = performance_df
            
            # Create title based on metric
            metric_names = {
                'Half_PPR': 'Half PPR Points',
                'Half_PPR_PPG': 'Half PPR Points Per Game',
                'VORP': 'Value Over Replacement Player',
                'VBD': 'Value Based Drafting Score'
            }
            title = f"{metric_names.get(metric, metric)} vs. ADP by Position"
            
            # Create scatter plot
            fig = px.scatter(
                filtered_df,
                x='ADP',
                y=metric,
                color='FantPos',
                hover_data=['Player', 'Team', 'G'],
                title=title,
            )
            
            # Update layout
            fig.update_layout(
                height=600,
                xaxis_title="Average Draft Position (ADP)",
                yaxis_title=metric_names.get(metric, metric),
                colorway=['#FF9999', '#99FF99', '#9999FF', '#FFFF99'],
            )
            
            # Add trend line
            fig.add_trace(
                go.Scatter(
                    x=filtered_df['ADP'],
                    y=filtered_df[metric],
                    mode='lines',
                    name='Trend',
                    line=dict(color='rgba(0,0,0,0.3)', dash='dash'),
                    showlegend=False
                )
            )
            
            return fig
        except Exception as e:
            # Return empty figure with error message
            fig = go.Figure()
            fig.update_layout(
                title="Error generating performance chart",
                annotations=[dict(text=str(e), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)]
            )
            return fig
    
    @app.callback(
        Output("top-performers-table", "children"),
        [Input("position-filter", "value"),
         Input("performance-metric", "value")]
    )
    def update_top_performers_table(position, metric):
        """Generate top performers table based on filters"""
        try:
            # Load performance data
            performance_df = pd.read_csv(os.path.join(processed_data_path, 'player_performance.csv'))
            
            # Filter by position if needed
            if position != "ALL":
                filtered_df = performance_df[performance_df['FantPos'] == position]
            else:
                filtered_df = performance_df
            
            # Sort by selected metric
            sorted_df = filtered_df.sort_values(metric, ascending=False).head(10)
            
            # Select columns to display
            display_cols = ['Player', 'FantPos', 'Team', metric, 'G', 'ADP']
            display_df = sorted_df[display_cols].copy()
            
            # Format metric values
            if metric in ['Half_PPR', 'Half_PPR_PPG', 'VORP', 'VBD']:
                display_df[metric] = display_df[metric].round(1)
            
            # Rename columns for display
            rename_map = {
                'Half_PPR': 'Half PPR',
                'Half_PPR_PPG': 'PPG',
                'VORP': 'VORP',
                'VBD': 'VBD',
                'FantPos': 'Pos',
                'G': 'Games'
            }
            display_df = display_df.rename(columns=rename_map)
            
            # Create data table
            table = dash_table.DataTable(
                data=display_df.to_dict('records'),
                columns=[{'name': rename_map.get(col, col), 'id': col} for col in display_cols],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ]
            )
            
            return html.Div([
                html.H5(f"Top 10 Performers by {rename_map.get(metric, metric)}"),
                table
            ])
        except Exception as e:
            return html.Div([
                html.H5("Error loading top performers data"),
                html.P(str(e))
            ])
    
    @app.callback(
        Output("over-under-performers-table", "children"),
        [Input("position-filter", "value")]
    )
    def update_over_under_performers_table(position):
        """Generate overperformers/underperformers table based on filters"""
        try:
            # Load expectation vs performance data
            expectation_df = pd.read_csv(os.path.join(processed_data_path, 'player_expectation_vs_performance.csv'))
            
            # Filter by position if needed
            if position != "ALL":
                filtered_df = expectation_df[expectation_df['FantPos'] == position]
            else:
                filtered_df = expectation_df
            
            # Get top overperformers and underperformers based on ADP vs actual rank
            over_df = filtered_df.sort_values('ADP_vs_Actual_Rank_Delta', ascending=False).head(5)
            under_df = filtered_df.sort_values('ADP_vs_Actual_Rank_Delta').head(5)
            
            # Select columns to display
            display_cols = ['Player', 'FantPos', 'Team', 'ADP', 'Overall_Rank', 'ADP_vs_Actual_Rank_Delta', 'Half_PPR']
            
            # Rename columns for display
            rename_map = {
                'FantPos': 'Pos',
                'Overall_Rank': 'Actual Rank',
                'ADP_vs_Actual_Rank_Delta': 'Rank Delta',
                'Half_PPR': 'Points'
            }
            
            # Format overperformers table
            over_display = over_df[display_cols].rename(columns=rename_map)
            over_table = dash_table.DataTable(
                data=over_display.to_dict('records'),
                columns=[{'name': rename_map.get(col, col), 'id': col} for col in display_cols],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ]
            )
            
            # Format underperformers table
            under_display = under_df[display_cols].rename(columns=rename_map)
            under_table = dash_table.DataTable(
                data=under_display.to_dict('records'),
                columns=[{'name': rename_map.get(col, col), 'id': col} for col in display_cols],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ]
            )
            
            return html.Div([
                html.H5("Top Overperformers (vs ADP)"),
                over_table,
                html.H5("Top Underperformers (vs ADP)", className="mt-4"),
                under_table
            ])
        except Exception as e:
            return html.Div([
                html.H5("Error loading performance vs expectations data"),
                html.P(str(e))
            ])
    
    # Position analysis tab callbacks
    @app.callback(
        Output("position-analysis-chart", "figure"),
        [Input("position-analysis-type", "value"),
         Input("position-focus", "value")]
    )
    def update_position_analysis_chart(analysis_type, position_focus):
        """Generate position analysis chart based on selected analysis type and position"""
        try:
            if analysis_type == "value":
                # Load VBD data
                vbd_df = pd.read_csv(os.path.join(processed_data_path, 'vbd_rankings.csv'))
                
                # Filter by position if needed
                if position_focus != "ALL":
                    filtered_df = vbd_df[vbd_df['FantPos'] == position_focus]
                else:
                    filtered_df = vbd_df
                
                # Create VBD distribution plot
                fig = px.box(
                    filtered_df,
                    x='FantPos',
                    y='VBD',
                    color='FantPos',
                    points='all',
                    hover_data=['Player', 'Team', 'Half_PPR'],
                    title='Value Over Baseline Distribution by Position',
                )
                
                # Add mean line
                for pos in filtered_df['FantPos'].unique():
                    pos_mean = filtered_df[filtered_df['FantPos'] == pos]['VBD'].mean()
                    fig.add_shape(
                        type="line",
                        x0=pos,
                        x1=pos,
                        y0=0,
                        y1=pos_mean,
                        line=dict(color="red", width=2, dash="dot"),
                    )
                
            elif analysis_type == "scarcity":
                # Load positional scarcity data
                position_files = {
                    'QB': 'QB_tiers.csv',
                    'RB': 'RB_tiers.csv',
                    'WR': 'WR_tiers.csv',
                    'TE': 'TE_tiers.csv'
                }
                
                # Determine which files to load based on position focus
                if position_focus != "ALL":
                    positions_to_load = [position_focus]
                else:
                    positions_to_load = ['QB', 'RB', 'WR', 'TE']
                
                # Load and combine data
                dfs = []
                for pos in positions_to_load:
                    try:
                        pos_df = pd.read_csv(os.path.join(processed_data_path, position_files[pos]))
                        dfs.append(pos_df)
                    except:
                        continue
                
                if not dfs:
                    raise Exception("No scarcity data available")
                
                combined_df = pd.concat(dfs)
                
                # Create line plot to show drop-offs
                fig = go.Figure()
                
                for pos in combined_df['FantPos'].unique():
                    pos_df = combined_df[combined_df['FantPos'] == pos].sort_values('Half_PPR', ascending=False).reset_index(drop=True)
                    
                    # Add player rank
                    pos_df['Rank'] = pos_df.index + 1
                    
                    # Plot points by rank
                    fig.add_trace(
                        go.Scatter(
                            x=pos_df['Rank'],
                            y=pos_df['Half_PPR'],
                            mode='lines+markers',
                            name=pos,
                            text=pos_df['Player'],
                            hovertemplate='%{text}<br>Rank: %{x}<br>Points: %{y:.1f}',
                        )
                    )
                
                # Add title and labels
                fig.update_layout(
                    title="Positional Scarcity - Points by Rank",
                    xaxis_title="Position Rank",
                    yaxis_title="Half PPR Points",
                )
                
            elif analysis_type == "tiers":
                # Load tier data
                if position_focus == "ALL":
                    # Default to RB tiers if all positions selected
                    position_focus = "RB"
                
                # Try to load tier data for selected position
                try:
                    tier_df = pd.read_csv(os.path.join(processed_data_path, f"{position_focus}_tiers.csv"))
                    tier_stats = pd.read_csv(os.path.join(processed_data_path, f"{position_focus}_tier_stats.csv"))
                except:
                    raise Exception(f"No tier data available for {position_focus}")
                
                # Create tier visualization
                fig = go.Figure()
                
                # Sort by points within each tier
                tier_df = tier_df.sort_values(['Tier', 'Half_PPR'], ascending=[True, False])
                
                # Get tier colors
                tier_colors = px.colors.qualitative.Set3[:len(tier_df['Tier'].unique())]
                tier_color_map = {tier: color for tier, color in zip(sorted(tier_df['Tier'].unique()), tier_colors)}
                
                # Add scatter plot for each tier
                for tier in sorted(tier_df['Tier'].unique()):
                    tier_subset = tier_df[tier_df['Tier'] == tier]
                    
                    fig.add_trace(
                        go.Box(
                            x=tier_subset['Tier'],
                            y=tier_subset['Half_PPR'],
                            name=tier,
                            boxpoints='all',
                            jitter=0.3,
                            pointpos=-1.8,
                            marker=dict(color=tier_color_map[tier]),
                            hovertext=tier_subset['Player'],
                            hoverinfo='text+y',
                        )
                    )
                
                # Add title and labels
                fig.update_layout(
                    title=f"{position_focus} Performance Tiers",
                    xaxis_title="Tier",
                    yaxis_title="Half PPR Points",
                )
            
            else:
                # Default to empty figure
                fig = go.Figure()
            
            # Update overall layout
            fig.update_layout(
                height=600,
                showlegend=True,
            )
            
            return fig
        except Exception as e:
            # Return empty figure with error message
            fig = go.Figure()
            fig.update_layout(
                title="Error generating position analysis",
                annotations=[dict(text=str(e), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)]
            )
            return fig
    
    @app.callback(
        Output("position-insights", "children"),
        [Input("position-analysis-type", "value"),
         Input("position-focus", "value")]
    )
    def update_position_insights(analysis_type, position_focus):
        """Generate position insights based on analysis type and position"""
        insights = {}
        
        # Define insights for each analysis type and position
        value_insights = {
            "ALL": [
                "RBs show the widest range of values, with elite options providing significant advantage",
                "WRs have more consistent value throughout the draft with less extreme outliers",
                "Elite QBs and TEs provide positional advantage, but the position value drops quickly",
                "Draft capital should be allocated based on positional scarcity and league structure"
            ],
            "QB": [
                "Elite QBs provide 5+ points per game advantage over replacement level",
                "Middle-tier QBs show similar production, making them better value picks",
                "Late-round QB strategy is viable if you miss on top options",
                "Dual-threat QBs typically provide the highest fantasy ceiling"
            ],
            "RB": [
                "Elite RBs provide the largest positional advantage in fantasy",
                "Significant drop-off after top 12-15 RBs makes early-round selections valuable",
                "Receiving-focused RBs offer more consistent production",
                "Backfield competition significantly impacts individual value"
            ],
            "WR": [
                "Elite WRs offer high floor and ceiling with consistent production",
                "The position shows less dramatic drop-offs compared to RB",
                "Target share and offensive environment strongly influence production",
                "Deep WR pool allows for value picks in middle rounds"
            ],
            "TE": [
                "Extreme top-heavy position with 2-3 elite options providing significant advantage",
                "Dramatic production cliff after top tier makes early investment or late flier optimal",
                "Middle-round TEs rarely return sufficient value on investment",
                "Red zone usage and route participation strongly correlate with fantasy success"
            ]
        }
        
        scarcity_insights = {
            "ALL": [
                "RB shows the steepest drop-off in production after the top tier",
                "WR has a more gradual decline in production throughout the ranks",
                "QB and TE show clear tiers with significant breaks between groups",
                "Positional scarcity should inform draft strategy and position prioritization"
            ],
            "QB": [
                "Clear top tier of 4-5 QBs before significant production drop",
                "Secondary tier of ~8 QBs with similar production levels",
                "Streaming options provide minimal point differential in later tiers",
                "Draft position strongly dictates optimal QB strategy"
            ],
            "RB": [
                "Steepest positional drop-off after top 12-15 options",
                "Elite RBs provide 8+ point advantage over replacement level",
                "Significant opportunity cost when missing on early RB selections",
                "Mid-round RBs with receiving work offer better floor"
            ],
            "WR": [
                "More gradual production decline compared to other positions",
                "Elite options still provide meaningful advantage over replacement",
                "Depth at position makes it viable to wait on WR selections",
                "Production cliffs are less pronounced but still apparent around WR24 and WR40"
            ],
            "TE": [
                "Extremely top-heavy position with significant advantage from elite options",
                "Major production cliff after top 3-4 options",
                "Late-round fliers at TE have similar production levels",
                "Middle tier offers minimal advantage over streaming options"
            ]
        }
        
        tier_insights = {
            "QB": [
                "Tier 1 QBs average 5+ PPG more than Tier 3 options",
                "Rushing production creates the biggest separation between tiers",
                "Tier 2 QBs offer the best value relative to ADP",
                "Dual-threat ability is common in higher tiers"
            ],
            "RB": [
                "Tier 1 RBs handle 20+ touches per game with receiving work",
                "Tier 2 RBs typically have either high volume or efficiency",
                "Tier 3 RBs are often in committee situations with limited upside",
                "Pass-catching RBs have more consistent production across tiers"
            ],
            "WR": [
                "Tier 1 WRs command 25%+ target share with high efficiency",
                "Tier 2 WRs offer either high volume or exceptional efficiency",
                "Tier 3 WRs show more volatility but still offer weekly upside",
                "Lower tiers show strong correlation with team passing volume"
            ],
            "TE": [
                "Elite TE tier produces like a WR2/WR3 with high consistency",
                "Second tier offers sporadic production with TD dependency",
                "Lower tiers show minimal differentiation in production",
                "Route participation percentage is a key differentiator between tiers"
            ]
        }
        
        # Select appropriate insights
        if analysis_type == "value":
            if position_focus in value_insights:
                insights = value_insights[position_focus]
            else:
                insights = value_insights["ALL"]
        elif analysis_type == "scarcity":
            if position_focus in scarcity_insights:
                insights = scarcity_insights[position_focus]
            else:
                insights = scarcity_insights["ALL"]
        elif analysis_type == "tiers":
            if position_focus in tier_insights:
                insights = tier_insights[position_focus]
            else:
                insights = tier_insights["QB"]  # Default to QB if not found
        
        # Create HTML elements
        return html.Div([
            html.H5(f"Key Insights for {position_focus if position_focus != 'ALL' else 'All Positions'}"),
            html.Ul([html.Li(insight) for insight in insights])
        ])
    
    # Draft strategy tab callbacks
    @app.callback(
        Output("draft-strategy-heatmap", "figure"),
        [Input("draft-pick-range", "value"),
         Input("team-building-strategy", "value")]
    )
    def update_draft_strategy_heatmap(pick_range, strategy):
        """Generate draft strategy heatmap based on pick range and strategy"""
        try:
            # Load position round stats
            position_round_stats = pd.read_csv(os.path.join(processed_data_path, 'position_round_stats.csv'))
            
            # Pivot data for heatmap
            heatmap_data = position_round_stats.pivot_table(
                index='Draft_Round',
                columns='Position',
                values='Success_Rate'
            ).reset_index()
            
            # Create heatmap figure
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data[['QB', 'RB', 'WR', 'TE']].values,
                x=['QB', 'RB', 'WR', 'TE'],
                y=heatmap_data['Draft_Round'],
                colorscale='Viridis',
                hoverongaps=False,
                showscale=True,
                colorbar=dict(title='Success Rate %'),
                text=heatmap_data[['QB', 'RB', 'WR', 'TE']].values,
                hovertemplate='Round: %{y}<br>Position: %{x}<br>Success Rate: %{z:.1f}%<extra></extra>',
            ))
            
            # Adjust based on team building strategy
            if strategy == 'rb_heavy':
                fig.add_annotation(
                    x='RB', y=1,
                    text="Priority",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    arrowsize=1,
                    arrowwidth=2,
                )
            elif strategy == 'wr_heavy':
                fig.add_annotation(
                    x='WR', y=1,
                    text="Priority",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    arrowsize=1,
                    arrowwidth=2,
                )
            elif strategy == 'elite_onesies':
                fig.add_annotation(
                    x='QB', y=1,
                    text="Priority",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    arrowsize=1,
                    arrowwidth=2,
                )
                fig.add_annotation(
                    x='TE', y=2,
                    text="Priority",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    arrowsize=1,
                    arrowwidth=2,
                )
            
            # Add title and labels
            fig.update_layout(
                title=f"Draft Strategy Heatmap - Success Rate by Position and Round<br><sub>Pick Range: {pick_range}, Strategy: {strategy}</sub>",
                xaxis_title="Position",
                yaxis_title="Draft Round",
                yaxis=dict(autorange="reversed"),  # Reverse y-axis to show round 1 at top
                height=600,
            )
            
            return fig
        except Exception as e:
            # Return empty figure with error message
            fig = go.Figure()
            fig.update_layout(
                title="Error generating draft strategy heatmap",
                annotations=[dict(text=str(e), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)]
            )
            return fig
    
    @app.callback(
        Output("draft-cheat-sheet", "children"),
        [Input("draft-pick-range", "value"),
         Input("team-building-strategy", "value")]
    )
    def update_draft_cheat_sheet(pick_range, strategy):
        """Generate draft cheat sheet based on pick range and strategy"""
        try:
            # Load strategy framework
            cheat_sheet_path = os.path.join(draft_strategy_path, 'draft_cheat_sheet.csv')
            if not os.path.exists(cheat_sheet_path):
                return html.Div([
                    html.H5("Draft Cheat Sheet Not Available"),
                    html.P("Please complete the draft strategy analysis first.")
                ])
            
            cheat_df = pd.read_csv(cheat_sheet_path)
            
            # Filter by pick range if specified
            if pick_range != "ALL" and pick_range in cheat_df['Pick_Range'].values:
                filtered_df = cheat_df[cheat_df['Pick_Range'] == pick_range]
            else:
                filtered_df = cheat_df
            
            # Apply strategy adjustments
            if strategy == 'rb_heavy':
                # Boost RB recommendations in early rounds
                for idx, row in filtered_df.iterrows():
                    if row['Round'] <= 4 and row['Primary_Position'] != 'RB':
                        filtered_df.at[idx, 'Primary_Position'] = 'RB'
            elif strategy == 'wr_heavy':
                # Boost WR recommendations in early rounds
                for idx, row in filtered_df.iterrows():
                    if row['Round'] <= 4 and row['Primary_Position'] != 'WR':
                        filtered_df.at[idx, 'Primary_Position'] = 'WR'
            elif strategy == 'elite_onesies':
                # Prioritize QB/TE in appropriate rounds
                for idx, row in filtered_df.iterrows():
                    if row['Round'] == 1:
                        filtered_df.at[idx, 'Primary_Position'] = 'QB'
                    elif row['Round'] == 2:
                        filtered_df.at[idx, 'Primary_Position'] = 'TE'
            
            # Select columns to display
            display_cols = ['Round', 'Overall_Picks', 'Primary_Position', 'Alternative_Positions',
                            'QB_Targets', 'RB_Targets', 'WR_Targets', 'TE_Targets']
            
            # Check available columns
            available_cols = [col for col in display_cols if col in filtered_df.columns]
            
            # Create data table
            table = dash_table.DataTable(
                data=filtered_df[available_cols].to_dict('records'),
                columns=[{'name': col.replace('_', ' '), 'id': col} for col in available_cols],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'minWidth': '100px', 'maxWidth': '180px'},
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ]
            )
            
            return html.Div([
                html.H5(f"Draft Cheat Sheet - {strategy.replace('_', ' ').title()} Strategy"),
                table
            ])
        except Exception as e:
            return html.Div([
                html.H5("Error Loading Draft Cheat Sheet"),
                html.P(str(e))
            ])
    
    @app.callback(
        Output("target-players-table", "children"),
        [Input("draft-pick-range", "value")]
    )
    def update_target_players_table(pick_range):
        """Generate target players table based on pick range"""
        try:
            # Load target player data if available
            target_files = {
                'QB': os.path.join(draft_strategy_path, 'QB_targets.csv'),
                'RB': os.path.join(draft_strategy_path, 'RB_targets.csv'),
                'WR': os.path.join(draft_strategy_path, 'WR_targets.csv'),
                'TE': os.path.join(draft_strategy_path, 'TE_targets.csv')
            }
            
            # Check which files exist
            available_targets = {}
            for pos, file_path in target_files.items():
                if os.path.exists(file_path):
                    available_targets[pos] = pd.read_csv(file_path)
            
            if not available_targets:
                return html.Div([
                    html.H5("Target Players Not Available"),
                    html.P("Please complete the player archetype and tier analysis first.")
                ])
            
            # Create tables for each position
            position_tables = []
            
            for pos, target_df in available_targets.items():
                # Filter by rounds relevant to the pick range
                if pick_range != "ALL" and "Round" in target_df.columns:
                    # Extract pick range numbers
                    if '-' in pick_range:
                        pick_start, pick_end = map(int, pick_range.split('-'))
                        
                        # Apply pick range filter by calculating relevant rounds
                        # For early picks (1-4), focus on rounds 1, 2, and later odd rounds
                        # For middle picks (5-8), focus on rounds 1, 3, and later even rounds
                        # For late picks (9-12), focus on rounds 2, 4, and later odd rounds
                        if pick_start <= 4:  # Early picks
                            round_filter = [1, 2, 5, 7, 9, 11]
                        elif pick_start <= 8:  # Middle picks
                            round_filter = [1, 3, 6, 8, 10, 12]
                        else:  # Late picks
                            round_filter = [2, 4, 5, 7, 9, 11]
                        
                        filtered_df = target_df[target_df['Round'].isin(round_filter)]
                    else:
                        filtered_df = target_df
                else:
                    filtered_df = target_df
                
                # Only create table if we have data
                if not filtered_df.empty:
                    # Select columns to display
                    display_cols = [col for col in ['Tier', 'Round', 'Target_Players', 'Avg_Points'] if col in filtered_df.columns]
                    
                    # Create data table
                    table = dash_table.DataTable(
                        data=filtered_df[display_cols].to_dict('records'),
                        columns=[{'name': col.replace('_', ' '), 'id': col} for col in display_cols],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto'},
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ]
                    )
                    
                    position_tables.append(html.Div([
                        html.H5(f"{pos} Target Players by Tier", className="mt-3"),
                        table
                    ]))
            
            return html.Div(position_tables)
        except Exception as e:
            return html.Div([
                html.H5("Error Loading Target Players"),
                html.P(str(e))
            ])