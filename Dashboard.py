#!/usr/bin/env python
# coding: utf-8

# In[2]:


import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px


# In[3]:


df = pd.read_csv('aircraft_parts_data_with_airlines_updated.csv', parse_dates=['Last Maintenance Date'])


# In[5]:


import dash_bootstrap_components as dbc


# In[6]:


import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px


# In[7]:


df = pd.read_csv('aircraft_parts_data_with_airlines_updated.csv', parse_dates=['Last Maintenance Date'])


# In[8]:


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# In[9]:


app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(
            html.H1("Real-Time Aircraft Parts Monitoring Dashboard", className='text-center text-primary mb-4'),
            width=12
        )
    ),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select Airline:"),
            dcc.Dropdown(
                id='airline-dropdown',
                options=[{'label': airline, 'value': airline} for airline in df['Airline'].unique()],
                value=df['Airline'].unique().tolist(),
                multi=True
            )
        ], width=6),
        
        dbc.Col([
            html.Label("Select Aircraft Type:"),
            dcc.Dropdown(
                id='aircraft-dropdown',
                options=[{'label': ac, 'value': ac} for ac in df['Aircraft Type'].unique()],
                value=df['Aircraft Type'].unique().tolist(),
                multi=True
            )
        ], width=6),
    ], className='mb-4'),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='risk-by-part')
        ], width=6),
        
        dbc.Col([
            dcc.Graph(id='condition-distribution')
        ], width=6),
    ]),
    
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(
                id='parts-table',
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records'),
                page_size=10,
                filter_action='native',
                sort_action='native',
                style_table={'overflowX': 'auto'},
                style_cell={
                    'minWidth': '150px', 'width': '150px', 'maxWidth': '150px',
                    'whiteSpace': 'normal'
                },
                style_data_conditional=[
                    {
                        'if': {
                            'filter_query': '{Predicted Damage} = 1',
                            'column_id': 'Predicted Damage'
                        },
                        'backgroundColor': 'tomato',
                        'color': 'white',
                        'fontWeight': 'bold'
                    },
                    {
                        'if': {
                            'filter_query': '{Risk of Failure} > 0.8',
                            'column_id': 'Risk of Failure'
                        },
                        'backgroundColor': 'red',
                        'color': 'white',
                        'fontWeight': 'bold'
                    }
                ]
            )
        ], width=12)
    ], className='mt-4'),
    
    # Refresh Interval
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # in milliseconds (60 seconds)
        n_intervals=0
    ),
], fluid=True)


# In[10]:


@app.callback(
    [Output('risk-by-part', 'figure'),
     Output('condition-distribution', 'figure'),
     Output('parts-table', 'data')],
    [Input('airline-dropdown', 'value'),
     Input('aircraft-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_dashboard(selected_airlines, selected_aircrafts, n):
    
    updated_df = pd.read_csv('aircraft_parts_data_with_airlines_updated.csv', parse_dates=['Last Maintenance Date'])
    
    filtered_df = updated_df[
        (updated_df['Airline'].isin(selected_airlines)) &
        (updated_df['Aircraft Type'].isin(selected_aircrafts))
    ]
    
    risk_summary = filtered_df.groupby('Part Name')['Risk of Failure'].mean().reset_index()
    fig1 = px.bar(risk_summary, x='Part Name', y='Risk of Failure',
                 title='Average Risk of Failure by Part',
                 labels={'Risk of Failure': 'Average Risk Score'},
                 hover_data={'Part Name': True, 'Risk of Failure': ':.2f'})
    fig1.update_layout(xaxis_tickangle=-45)
    
    condition_counts = filtered_df['Current Condition'].value_counts().reset_index()
    condition_counts.columns = ['Current Condition', 'Count']
    fig2 = px.pie(condition_counts, names='Current Condition', values='Count',
                 title='Distribution of Current Conditions')
    
    table_data = filtered_df.to_dict('records')
    
    return fig1, fig2, table_data

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




