"""
Fantasy Football Analysis Dashboard
This module implements a Dash application to visualize fantasy football analysis results.
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
import sys
from pathlib import Path

# Add parent directory to path to import project modules
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

# Import project modules
from src.data.data_loader import load_config
from layouts import create_layout
from callbacks import register_callbacks

# Load configuration
config = load_config()

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    title="Fantasy Football Analysis Dashboard"
)

# Create layout
app.layout = create_layout()

# Register callbacks
register_callbacks(app)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)