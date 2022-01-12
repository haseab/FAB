import sys

import numpy as np
import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash import (Dash, Input,  # pip install dash (version 2.0.0 or higher)
                  Output, dash_table, dcc, html)
from jupyter_dash import JupyterDash


def run(trader, illustrator):
    current_positions = trader.get_current_trade_progress()
    portfolio_pnl = (current_positions['position share']*current_positions['pnl (%)']).sum()

    def graph_trading_history_wrapper():
        fig = illustrator.graph_trading_history(trader, "2022-01-11 10:00:00")
        return fig
    
    def get_pie_chart():
        fig = illustrator.current_trades_portfolio(current_positions)
        return fig 
 
    
    app = JupyterDash(__name__)

    app.layout = html.Div([

        html.H1(children = "FAB Realtime Dashboard",
                style={'text-align': 'center',
                       }),
        
        html.Div([
        
        html.H2('Current Positions', style={'text-align': 'center'}),

        dcc.Graph(id='pie_chart',
                  figure=get_pie_chart(),
                  style={'width': '45vh',
                         'height': '45vh',
                         'padding-left': "250px"}
                  ),

        dash_table.DataTable(
            id='table',
            columns = [{"name": i, "id": i} for i in current_positions.columns],
            data = current_positions.to_dict('records'),
            fill_width=False,
            style_header={ 'border': '1px solid black' },
            style_data={'whiteSpace': 'normal',
                        'height': 'auto',
                        'border': '1px solid black'
                        },
        ),

        ], className = 'column-1'),

        html.Div([
        html.H2('Trading History', style={'text-align': 'center'}),
        dcc.Graph(id='net_worth', 
                  figure=graph_trading_history_wrapper(), 
                  style={'width': '90vh', 
                        'height': '45vh',
                        }
                  ),

        html.H1(f"Portfolio Profit: {round(portfolio_pnl,2)}%",
                style={'color':'red'}),



        
        ], className='column-2'),

        


        ], className='row', 
           style = {"padding-bottom":'100px',
                    'font-family': "Verdana"})

    # Connect the Plotly graphs with Dash Components


        

    app.run_server(debug=True)
