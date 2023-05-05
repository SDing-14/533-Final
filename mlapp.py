from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime, date, timedelta
import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime
import os
import eikon as ek
# import refinitiv.dataplatform.eikon as ek
import refinitiv.data as rd
import blotter as blt
import ledger as lg
import ml as ml
from dash import html
import base64
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

percentage = dash_table.FormatTemplate.percentage(3)

image_filename = 'IMG_0614.jpg'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())



controls = dbc.Card(
    [
        dbc.Row(html.Button('QUERY Refinitiv', id='run-query', n_clicks=0)),
        dbc.Row([
            html.H5('Asset:',
                    style={'display': 'inline-block', 'margin-right': 20}),
            dcc.Input(id='asset', type='text', value="IVV",
                      style={'display': 'inline-block',
                             'border': '1px solid black'}),
            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("Î±1"), html.Th("n1")])),
                    html.Tbody([
                        html.Tr([
                            html.Td(
                                dbc.Input(
                                    id='alpha1',
                                    type='number',
                                    value=-0.01,
                                    max=1,
                                    min=-1,
                                    step=0.01,
                                    style={'width':'auto'}
                                )
                            ),
                            html.Td(
                                dcc.Input(
                                    id='n1',
                                    type='number',
                                    value=3,
                                    min=1,
                                    step=1,
                                    style={'width':'auto'}
                                )
                            )
                        ])
                    ])
                ],
                bordered=True
            ),
            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("Î±2"), html.Th("n2")])),
                    html.Tbody([
                        html.Tr([
                            html.Td(
                                dbc.Input(
                                    id='alpha2',
                                    type='number',
                                    value=0.01,
                                    max=1,
                                    min=-1,
                                    step=0.01,
                                    style={'width':'auto'}
                                )
                            ),
                            html.Td(
                                dcc.Input(
                                    id='n2',
                                    type='number',
                                    value=5,
                                    min=1,
                                    step=1,
                                    style={'width':'auto'}
                                )
                            )
                        ])
                    ])
                ],
                bordered=True
            )
        ]),
        dbc.Row(html.Button('UPDATE BLOTTER', id='update-blotter', n_clicks=0)),
        dbc.Row([
            dcc.DatePickerRange(
                id='refinitiv-date-range',
                min_date_allowed = date(2015, 1, 1),
                max_date_allowed = date(2023, 4, 13),
                # max_date_allowed = datetime.now(),
                start_date = datetime.date(
                    # datetime.now() - timedelta(days=3*365)
                    datetime(2023, 4, 13) - timedelta(days=250)
                ),
                end_date = date(2023, 4, 13) #datetime.now().date()
            )
        ])
    ],
    body=True,
)

app.layout = dbc.Container(
    [   html.H2('Created/Modified by: Xiaokuan Zhao, Rebecca Xiao, Xinyang Ding',
                style={'font-size': '36px', 'color': '#150',
                       'font-family': 'Times New Roman, sans-serif', 'text-align': 'center'}),
        dbc.Row(
            [
                dbc.Col(controls, md=4, align="center"),
                dbc.Col(
                    # Put your reactive graph here as an image!
                    html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), alt="image",
                             style = {'width': '50%', 'height': '50%'}),
                    md = 8,
                    align="center"
                )
            ],
            align="center",
        ),
        html.H2('Trade Blotter:', style={'fontSize': '28px', 'color': '#150',
                       'fontFamily': 'Times New Roman, sans-serif'}),
        dash_table.DataTable(id = "blotter", page_size=10),
        html.H2('Trade Ledger:', style={'fontSize': '28px', 'color': '#150',
                       'fontFamily': 'Times New Roman, sans-serif'}),
        dash_table.DataTable(id = "ledger",page_size=10),
        html.H2('ML Success Compared:', style={'fontSize': '28px', 'color': '#150',
                    'fontFamily': 'Times New Roman, sans-serif'}),
        dash_table.DataTable(id = "compare_success",page_size=10),
        html.Div([dcc.Graph(id='hist_return')]),
        html.Div([dcc.Graph(id='ml_return')]),
        html.Div([
            html.H2('App Purpose:', style={'fontSize': '28px', 'color': '#150',
                       'fontFamily': 'Times New Roman, sans-serif'}),
            html.P('Hi, this is our trading app.'),
            html.P('We did not choose either free stocks or a vol trader strategy. Instead, our trading app is an improvement of the current a1-a2-n1-n2 strategy. To be more specific, we adapted four features from the dataset and used a decision tree to distinguish from 1 and (0 and -1) groups as an indicator of whether we want to initiate buying on one certain day. This approach will stop us from entering the market on some specific dates, in this way, we can avoid potential big losses and also retain a stable profit.'),
            html.P('The reason why we use the decision tree model is it is a simple classification strategy and it is easier to explain, especially how it uses the features. We can easily draw a graph to show how the features are used.'),
            html.P('The features we used are the VIX index, DXY Curncy, SPXSFRCS Index, and IVV AU Equity. The reason why we choose these is as follows. First, those index covers different aspects of economics, like volatility and currency value. Next, these features process some ideal properties such as they are there is no significant multilinearity as will be an issue in linear regression and many regression models. The way we filter the features are based heat map.'),
            html.P('Because our features are only four dimensions, there is no need to use dimensional reduction. If we do need to, we can use PCA to reduce dimension.'),
            html.P('Our look back window size is 10, and we run on the last 100 days of data. 10 days of window size correspond to the last two week\'s market data, and we only would want the last 100 days (roughly half a trading year) to train our model because we want our model to only adapt to the most recent macro economic environment.'),
            html.P('As for the application of Hoeffding inequation, we set our X to be the achieved volatility collected from our revised algorithm, the Miu to be the achieved volatility of the \'dumb\' method and alpha and beta to be 1 times the average difference. So, our X is roughly 0.06% per trade and Miu is 0.11% per trade, and we put everything in our Hoeffding equation and find the result that if the annualized volatility difference of our model and the \'dumb\' model is less than 4.98%, then our model is no longer effective and need to be halted.'),
        ], style={'margin': '50px 0'})
    ],
    fluid=True
)

@app.callback(
    Output("blotter","data"),
    [Input("update-blotter", "n_clicks"),
     Input("run-query", "n_clicks")],
    [State('refinitiv-date-range','start_date'),
     State('refinitiv-date-range', 'end_date'),
     State('alpha1', 'value'),
     State('n1', 'value'),
     State('alpha2', 'value'),
     State('n2', 'value'),
     State('asset', 'value'),
     ],
    prevent_initial_call = True
)

def combined_callback(btn1_clicks, btn2_clicks, start_date, end_date, alpha1, n1, alpha2, n2, asset_id):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if "run-query" in changed_id:
        blt.query_data(start_date, end_date, asset_id)
        blotter = blt.make_blotter(alpha1, n1, alpha2, n2)

        return blotter.to_dict('records')

    elif "update-blotter" in changed_id:
        blotter = blt.make_blotter(alpha1, n1, alpha2, n2)

        return blotter.to_dict('records')

@app.callback(
    Output("ledger", "data"),
    Input("blotter", "data"),
    prevent_initial_call = True
)

def create_ledger(blotter):
    ledger = lg.create_ledger(blotter)
    return ledger.to_dict('records')

@app.callback(
    Output("compare_success", "data"),
    Input("ledger", "data"),
    prevent_initial_call = True
)

def train_ml_result(ledger):
    compare_success = ml.train_and_result(ledger)
    return compare_success.to_dict('records')


@app.callback(
    Output("hist_return", "figure"),
    #  Output("hist_alpha", "num"),
    Input("ledger", "data"),
    prevent_initial_call = True
)

def hist_return(ledger):
    x, y, alpha1, beta1 = ml.ledger_return(ledger)
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', name='Original Return'))
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(x=x, y=p(x), mode='lines', name='Trendline'))
    equation = f'y = {z[0]:.2f}x + {z[1]:.2f}' + '\nReturn = beta * x + alpha'

    # fig.add_trace(go.Scatter(x=x, y=y, name='Original Return'))

    # x, y, alpha2, beta2 = ml.ml_return(ledger)
    # fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='ML Return', line=dict(color='blue')))

    # fig.add_annotation(x=1.1, y=0.9, text=f"Original Alpha: {alpha1:.2f}", showarrow=False)
    # fig.add_annotation(x=1.1, y=0.8, text=f"Original Beta: {beta1:.2f}", showarrow=False)
    annotations = [
        dict(xref='paper', yref='paper', x=0.8, y=0.17, xanchor='left', yanchor='top', text=f"Original Alpha: {alpha1:.5f}", showarrow=False),
        dict(xref='paper', yref='paper', x=0.8, y=0.1, xanchor='left', yanchor='top', text=f"Original Beta: {beta1:.5f}", showarrow=False),
    ]

    fig.update_layout(
        title="IVV Return vs. Oringinal Strategy Return",
        xaxis_title="IVV/Market Return",
        yaxis_title="Original Strategy Return",
        annotations=annotations
    )
    fig.add_annotation(x=0.02, y=0.04, xanchor='left', yanchor='top',
                text=equation, showarrow=False)

    return fig




@app.callback(
    Output("ml_return", "figure"),
    Input("ledger", "data"),
    prevent_initial_call = True
)

def ml_return(ledger):
    x, y, alpha2, beta2 = ml.ml_return(ledger)
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', name='ML Return'))
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(x=x, y=p(x), mode='lines', name='Trendline'))
    equation = f'y = {z[0]:.2f}x + {z[1]:.2f}' + '\nReturn = beta * x + alpha'


    annotations = [
        dict(xref='paper', yref='paper', x=0.8, y=0.17, xanchor='left', yanchor='top', text=f"ML Alpha: {alpha2:.5f}", showarrow=False),
        dict(xref='paper', yref='paper', x=0.8, y=0.1, xanchor='left', yanchor='top', text=f"ML Beta: {beta2:.5f}", showarrow=False),
    ]


    fig.update_layout(
        title="IVV Return vs. ML Strategy Return",
        xaxis_title="IVV/Market Return",
        yaxis_title="ML Strategy Return",
        annotations=annotations
    )
    fig.add_annotation(x=0.02, y=0.04, xanchor='left', yanchor='top',
                text=equation, showarrow=False)

    return fig



if __name__ == '__main__':
    app.run_server(debug=True, port = 8000)
