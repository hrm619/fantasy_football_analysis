"""
Dashboard layouts module
Defines the layout components for the Fantasy Football Analysis Dashboard
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path to import project modules
parent_dir = str(Path(__file__).resolve().parent.parent)
import sys
sys.path.append(parent_dir)

# Import project modules
from src.data.data_loader import load_config

# Load configuration
config = load_config()

# Load processed data paths
processed_data_path = config['data_paths']['processed_data']
draft_strategy_path = os.path.join(processed_data_path, 'draft_strategy')

def create_layout():
    """
    Create the dashboard layout
    
    Returns:
        dash.html.Div: The dashboard layout
    """
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Fantasy Football Analysis Dashboard", className="text-center my-4"),
                html.H5("2025 Season Draft Preparation", className="text-center mb-4"),
            ], width=12)
        ]),
        
        # Navigation tabs
        dbc.Row([
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(label="Dashboard Overview", tab_id="tab-overview"),
                    dbc.Tab(label="Player Performance", tab_id="tab-performance"),
                    dbc.Tab(label="Position Analysis", tab_id="tab-position"),
                    dbc.Tab(label="Draft Strategy", tab_id="tab-draft"),
                    dbc.Tab(label="Player Archetypes", tab_id="tab-archetypes"),
                    dbc.Tab(label="Team Context", tab_id="tab-team"),
                ], id="tabs", active_tab="tab-overview"),
            ], width=12)
        ]),
        
        # Content container
        dbc.Row([
            dbc.Col([
                html.Div(id="tab-content", className="mt-4")
            ], width=12)
        ]),
        
        # Footer
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P("Fantasy Football Analysis Dashboard - Created with Dash", className="text-center text-muted"),
            ], width=12)
        ]),
        
    ], fluid=True)

def create_overview_layout():
    """
    Create the overview tab layout
    
    Returns:
        dash.html.Div: The overview layout
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Dashboard Overview"),
                    dbc.CardBody([
                        html.P("This dashboard provides comprehensive fantasy football analysis for the 2025 season."),
                        html.P("Navigate through the tabs to explore different aspects of the analysis:"),
                        html.Ul([
                            html.Li("Player Performance: Analyze player performance and fantasy production"),
                            html.Li("Position Analysis: Compare value across positions"),
                            html.Li("Draft Strategy: Optimize your draft strategy"),
                            html.Li("Player Archetypes: Explore player types and characteristics"),
                            html.Li("Team Context: Understand team context impact"),
                        ]),
                    ])
                ]),
            ], width=12),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Key Insights"),
                    dbc.CardBody([
                        html.Div(id="key-insights")
                    ])
                ]),
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Draft Strategy Summary"),
                    dbc.CardBody([
                        html.Div(id="draft-summary")
                    ])
                ]),
            ], width=6),
        ], className="mt-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Position Value Overview"),
                    dbc.CardBody([
                        dcc.Graph(id="position-value-overview")
                    ])
                ]),
            ], width=12),
        ], className="mt-4"),
    ])

def create_performance_layout():
    """
    Create the player performance tab layout
    
    Returns:
        dash.html.Div: The performance layout
    """
    try:
        # Load performance data
        performance_df = pd.read_csv(os.path.join(processed_data_path, 'player_performance.csv'))
        
        # Create position options
        position_options = [{'label': 'All Positions', 'value': 'ALL'}]
        for pos in ['QB', 'RB', 'WR', 'TE']:
            position_options.append({'label': pos, 'value': pos})
        
        # Create default performance plot
        fig = px.scatter(
            performance_df,
            x='ADP',
            y='Half_PPR',
            color='FantPos',
            hover_data=['Player', 'Team'],
            title='Fantasy Points vs. ADP',
        )
        fig.update_layout(height=600)
    except Exception as e:
        # Create empty layout on error
        print(f"Error loading performance data: {e}")
        position_options = [{'label': 'All Positions', 'value': 'ALL'}]
        fig = go.Figure()
        fig.update_layout(
            title="Error loading performance data",
            annotations=[dict(text=str(e), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)]
        )
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Player Performance Filters"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Select Position"),
                                dcc.Dropdown(
                                    id="position-filter",
                                    options=position_options,
                                    value="ALL",
                                    clearable=False
                                ),
                            ], width=6),
                            
                            dbc.Col([
                                html.Label("Performance Metric"),
                                dcc.Dropdown(
                                    id="performance-metric",
                                    options=[
                                        {'label': 'Half PPR Points', 'value': 'Half_PPR'},
                                        {'label': 'Points Per Game', 'value': 'Half_PPR_PPG'},
                                        {'label': 'VORP', 'value': 'VORP'},
                                        {'label': 'VBD', 'value': 'VBD'},
                                    ],
                                    value="Half_PPR",
                                    clearable=False
                                ),
                            ], width=6),
                        ]),
                    ])
                ]),
            ], width=12),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Player Performance Chart"),
                    dbc.CardBody([
                        dcc.Graph(id="performance-chart", figure=fig)
                    ])
                ]),
            ], width=12),
        ], className="mt-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Top Performers Table"),
                    dbc.CardBody([
                        html.Div(id="top-performers-table")
                    ])
                ]),
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Overperformers/Underperformers"),
                    dbc.CardBody([
                        html.Div(id="over-under-performers-table")
                    ])
                ]),
            ], width=6),
        ], className="mt-4"),
    ])

def create_position_layout():
    """
    Create the position analysis tab layout
    
    Returns:
        dash.html.Div: The position layout
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Position Analysis Controls"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Analysis Type"),
                                dcc.Dropdown(
                                    id="position-analysis-type",
                                    options=[
                                        {'label': 'Position Value Comparison', 'value': 'value'},
                                        {'label': 'Positional Scarcity', 'value': 'scarcity'},
                                        {'label': 'Position Tiers', 'value': 'tiers'},
                                    ],
                                    value="value",
                                    clearable=False
                                ),
                            ], width=6),
                            
                            dbc.Col([
                                html.Label("Position Focus"),
                                dcc.Dropdown(
                                    id="position-focus",
                                    options=[
                                        {'label': 'All Positions', 'value': 'ALL'},
                                        {'label': 'QB', 'value': 'QB'},
                                        {'label': 'RB', 'value': 'RB'},
                                        {'label': 'WR', 'value': 'WR'},
                                        {'label': 'TE', 'value': 'TE'},
                                    ],
                                    value="ALL",
                                    clearable=False
                                ),
                            ], width=6),
                        ]),
                    ])
                ]),
            ], width=12),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Position Analysis Chart"),
                    dbc.CardBody([
                        dcc.Graph(id="position-analysis-chart")
                    ])
                ]),
            ], width=12),
        ], className="mt-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Position Value Insights"),
                    dbc.CardBody([
                        html.Div(id="position-insights")
                    ])
                ]),
            ], width=12),
        ], className="mt-4"),
    ])

def create_draft_layout():
    """
    Create the draft strategy tab layout
    
    Returns:
        dash.html.Div: The draft layout
    """
    # Load draft strategy data if available
    try:
        strategy_df = pd.read_csv(os.path.join(draft_strategy_path, 'strategy_framework.csv'))
        
        # Create pick strategy options
        pick_options = []
        for pick_range in strategy_df['Pick_Range'].unique():
            pick_options.append({'label': f"Picks {pick_range}", 'value': pick_range})
    except Exception as e:
        print(f"Error loading draft strategy data: {e}")
        pick_options = [{'label': 'All Picks', 'value': 'ALL'}]
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Draft Strategy Controls"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Draft Position Range"),
                                dcc.Dropdown(
                                    id="draft-pick-range",
                                    options=pick_options,
                                    value=pick_options[0]['value'] if pick_options else None,
                                    clearable=False
                                ),
                            ], width=6),
                            
                            dbc.Col([
                                html.Label("Team Building Strategy"),
                                dcc.Dropdown(
                                    id="team-building-strategy",
                                    options=[
                                        {'label': 'Balanced Approach', 'value': 'balanced'},
                                        {'label': 'RB Heavy', 'value': 'rb_heavy'},
                                        {'label': 'WR Heavy', 'value': 'wr_heavy'},
                                        {'label': 'Elite QB/TE Focus', 'value': 'elite_onesies'},
                                    ],
                                    value="balanced",
                                    clearable=False
                                ),
                            ], width=6),
                        ]),
                    ])
                ]),
            ], width=12),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Draft Strategy Heatmap"),
                    dbc.CardBody([
                        dcc.Graph(id="draft-strategy-heatmap")
                    ])
                ]),
            ], width=12),
        ], className="mt-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Draft Cheat Sheet"),
                    dbc.CardBody([
                        html.Div(id="draft-cheat-sheet")
                    ])
                ]),
            ], width=12),
        ], className="mt-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Target Players by Round"),
                    dbc.CardBody([
                        html.Div(id="target-players-table")
                    ])
                ]),
            ], width=12),
        ], className="mt-4"),
    ])

def create_archetypes_layout():
    """
    Create the player archetypes tab layout
    
    Returns:
        dash.html.Div: The archetypes layout
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Player Archetype Controls"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Position"),
                                dcc.Dropdown(
                                    id="archetype-position",
                                    options=[
                                        {'label': 'QB', 'value': 'QB'},
                                        {'label': 'RB', 'value': 'RB'},
                                        {'label': 'WR', 'value': 'WR'},
                                        {'label': 'TE', 'value': 'TE'},
                                    ],
                                    value="RB",
                                    clearable=False
                                ),
                            ], width=6),
                            
                            dbc.Col([
                                html.Label("Visualization Type"),
                                dcc.Dropdown(
                                    id="archetype-viz-type",
                                    options=[
                                        {'label': 'Archetype Clusters', 'value': 'clusters'},
                                        {'label': 'Radar Charts', 'value': 'radar'},
                                        {'label': 'Performance by Archetype', 'value': 'performance'},
                                    ],
                                    value="clusters",
                                    clearable=False
                                ),
                            ], width=6),
                        ]),
                    ])
                ]),
            ], width=12),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Player Archetype Visualization"),
                    dbc.CardBody([
                        dcc.Graph(id="archetype-chart")
                    ])
                ]),
            ], width=12),
        ], className="mt-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Archetype Characteristics"),
                    dbc.CardBody([
                        html.Div(id="archetype-characteristics")
                    ])
                ]),
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Example Players"),
                    dbc.CardBody([
                        html.Div(id="archetype-examples")
                    ])
                ]),
            ], width=6),
        ], className="mt-4"),
    ])

def create_team_layout():
    """
    Create the team context tab layout
    
    Returns:
        dash.html.Div: The team layout
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Team Context Controls"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Analysis Focus"),
                                dcc.Dropdown(
                                    id="team-analysis-focus",
                                    options=[
                                        {'label': 'Team Performance Correlation', 'value': 'correlation'},
                                        {'label': 'Opportunity Share', 'value': 'opportunity'},
                                        {'label': 'Offensive Line Impact', 'value': 'oline'},
                                    ],
                                    value="correlation",
                                    clearable=False
                                ),
                            ], width=6),
                            
                            dbc.Col([
                                html.Label("Position Filter"),
                                dcc.Dropdown(
                                    id="team-position-filter",
                                    options=[
                                        {'label': 'All Positions', 'value': 'ALL'},
                                        {'label': 'QB', 'value': 'QB'},
                                        {'label': 'RB', 'value': 'RB'},
                                        {'label': 'WR', 'value': 'WR'},
                                        {'label': 'TE', 'value': 'TE'},
                                    ],
                                    value="ALL",
                                    clearable=False
                                ),
                            ], width=6),
                        ]),
                    ])
                ]),
            ], width=12),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Team Context Visualization"),
                    dbc.CardBody([
                        dcc.Graph(id="team-context-chart")
                    ])
                ]),
            ], width=12),
        ], className="mt-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Team Context Insights"),
                    dbc.CardBody([
                        html.Div(id="team-context-insights")
                    ])
                ]),
            ], width=12),
        ], className="mt-4"),
    ])