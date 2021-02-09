import dash
import dash_core_components as dcc
import dash_html_components as html


import plotly.graph_objects as go

from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Coin Web'),

    dcc.Dropdown(id='secim',
        options=[

            {'label': 'BTC', 'value': 'BTC'},
            {'label': 'LTC', 'value': 'LTC'},
            {'label': 'ADA', 'value': 'ADA'},

            {'label': 'DOGE', 'value': 'DOGE'},
            {'label': 'BCN', 'value': 'BCN'},
            {'label': 'SC', 'value': 'SC'},

            {'label': 'XVG', 'value': 'XVG'},
            {'label': 'DGB', 'value': 'DGB'},
            {'label': 'NLC2', 'value': 'NLC2'},


        ],

        value='LTC',
        multi=False
    ),

    html.Div(children='''
        Coin Örnek Uygulama
    '''),

    dcc.Graph(
        id='Sonuc',

        ),
    dcc.Interval(
            id='interval-component',
            interval=1*60000, # in milliseconds
            n_intervals=0),

    dcc.Graph(
        id='data'
    ),

dcc.Graph(
        id='MACD'
    ),

dcc.Graph(
        id='Table1'
    ),

dcc.Graph(
        id='OBV'
    ),

])
@app.callback(Output('Sonuc','figure'),[Input('secim','value'),Input('interval-component', 'n_intervals')])
def mumgraph(secilen,n):

    from pandas_datareader import data as pdr
    import yfinance as yf
    hisse = secilen + '-USD'

    df = pdr.get_data_yahoo(hisse, start='2020-01-01', end='2021-02-07')
    df.reset_index(inplace = True)



    df['3Sma'] = df['Close'].rolling(window=3).mean()
    df['7Sma'] = df['Close'].rolling(window=7).mean()
    df['14Sma'] = df['Close'].rolling(window=14).mean()
    df['20Sma'] = df['Close'].rolling(window=20).mean()

    # Artış Hesaplanması
    df['Artıs'] = (df['Close'] - df['Open']) / df['Open']

    # Bollinger Band Hesaplanması
    df['stdev'] = df['Close'].rolling(window=20).std()
    df['lower_band'] = df['20Sma'] - 2 * df['stdev']
    df['upper_band'] = df['20Sma'] + 2 * df['stdev']

    # Keltner Band Hesaplanması
    df['TR'] = abs(df['High'] - df['Low'])
    df['ATR'] = df['TR'].rolling(window=20).mean()
    df['Lower_Kelter'] = df['20Sma'] - (df['ATR'] * 1.5)
    df['Upper_Kelter'] = df['20Sma'] + (df['ATR'] * 1.5)

    # MACD Hesaplanması

    df['9Sma'] = df['Close'].rolling(window=9).mean()
    df['12Sma'] = df['Close'].rolling(window=12).mean()
    df['26Sma'] = df['Close'].rolling(window=26).mean()

    df['MACD'] = df['26Sma'] - df['12Sma']

    efes_df = df

    mumlar = go.Candlestick(x=efes_df['Date'],open=efes_df['Open'],high=efes_df['High'],low=efes_df['Low'],close=efes_df['Close'])

    upper_band=go.Scatter(x=efes_df['Date'],y=efes_df['upper_band'],name='Upper Bollinger',line={'color':'red'})
    lower_band=go.Scatter(x=efes_df['Date'],y=efes_df['lower_band'],name='Lower Bollinger',line={'color':'red'})

    upper_kelter=go.Scatter(x=efes_df['Date'],y=efes_df['Upper_Kelter'],name='Upper Kelter',line={'color':'blue'})
    lower_kelter=go.Scatter(x=efes_df['Date'],y=efes_df['Lower_Kelter'],name='Lower Kelter',line={'color':'blue'})



    layout = go.Layout(
        title=hisse,
        height=1000
    )

    fig = go.Figure(data=[mumlar,lower_band,upper_band,upper_kelter,lower_kelter],layout=layout)


    return fig
@app.callback(Output('data','figure'),[Input('secim','value')])
def RSIGraph(veri):

    from pandas_datareader import data as pdr
    import yfinance as yf
    hisse = veri+ '-USD'

    df = pdr.get_data_yahoo(hisse, start='2020-01-01', end='2021-02-07')
    df.reset_index(inplace = True)

    delta = df['Close'].diff()
    delta.dropna(inplace=True)

    pos = delta.copy()
    neg = delta.copy()
    pos[pos < 0] = 0
    neg[neg > 0] = 0

    days = 14

    gain = pos.rolling(window=days).mean()
    loss = abs(neg.rolling(window=days).mean())

    RS = gain / loss

    RSI = 100 - (100 / (1 + RS))

    df['RSİ'] = RSI

    df['3Sma'] = df['Close'].rolling(window=3).mean()
    df['7Sma'] = df['Close'].rolling(window=7).mean()
    df['14Sma'] = df['Close'].rolling(window=14).mean()
    df['20Sma'] = df['Close'].rolling(window=20).mean()

    # Artış Hesaplanması
    df['Artıs'] = (df['Close'] - df['Open']) / df['Open']

    # Bollinger Band Hesaplanması
    df['stdev'] = df['Close'].rolling(window=20).std()
    df['lower_band'] = df['20Sma'] - 2 * df['stdev']
    df['upper_band'] = df['20Sma'] + 2 * df['stdev']

    # Keltner Band Hesaplanması
    df['TR'] = abs(df['High'] - df['Low'])
    df['ATR'] = df['TR'].rolling(window=20).mean()
    df['Lower_Kelter'] = df['20Sma'] - (df['ATR'] * 1.5)
    df['Upper_Kelter'] = df['20Sma'] + (df['ATR'] * 1.5)

    # MACD Hesaplanması

    df['9Sma'] = df['Close'].rolling(window=9).mean()
    df['12Sma'] = df['Close'].rolling(window=12).mean()
    df['26Sma'] = df['Close'].rolling(window=26).mean()

    df['MACD'] = df['26Sma'] - df['12Sma']

    layout = go.Layout(
        title=hisse,
        height=600
    )

    trace0 = go.Scatter(x=df['Date'], y=df['RSİ'], mode='lines')
    fig = go.Figure(trace0,layout=layout)



    return fig
@app.callback(Output('MACD','figure'),[Input('secim','value')])
def SMAGraph(veri):

    from pandas_datareader import data as pdr
    import yfinance as yf
    hisse = veri+ '-USD'

    df = pdr.get_data_yahoo(hisse, start='2020-01-01', end='2021-02-07')
    df.reset_index(inplace = True)

    df['9Sma'] = df['Close'].rolling(window=9).mean()
    df['12Sma'] = df['Close'].rolling(window=12).mean()
    df['26Sma'] = df['Close'].rolling(window=26).mean()

    df['MACD'] = df['26Sma'] - df['12Sma']


    import plotly.express as px

    fig = px.line(df, x="Date", y=["Close", "9Sma", "12Sma", '26Sma'])

    return fig
@app.callback(Output('Table1','figure'),[Input('secim','value')])
def Tablo(veri):

    from pandas_datareader import data as pdr
    import yfinance as yf
    hisse = veri+ '-USD'

    df = pdr.get_data_yahoo(hisse, start='2020-01-01', end='2021-02-07')
    df.reset_index(inplace = True)

    df['Hisse'] = veri

    df['9Sma'] = df['Close'].rolling(window=9).mean()
    df['12Sma'] = df['Close'].rolling(window=12).mean()
    df['26Sma'] = df['Close'].rolling(window=26).mean()

    df['MACD'] = df['26Sma'] - df['12Sma']



    dfx = df[['Date','Close','Open','High','Low','Volume','Hisse']]

    decimals = 4

    dfx['Close'] = dfx['Close'].apply(lambda x: round(x, decimals))
    dfx['Open'] = dfx['Open'].apply(lambda x: round(x, decimals))
    dfx['High'] = dfx['High'].apply(lambda x: round(x, decimals))
    dfx['Low'] = dfx['Low'].apply(lambda x: round(x, decimals))
    dfx['Volume'] = dfx['Volume'].apply(lambda x: round(x, 2))

    dfy = dfx.tail(10)





    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(dfy.columns),
        ),
        cells=dict(values=[
            dfy['Date'],
            dfy['Close'],
            dfy['Open'],
            dfy['High'],
            dfy['Low'],
            dfy['Volume'],
            dfy['Hisse']
            ]))])
    return fig
@app.callback(Output('OBV','figure'),[Input('secim','value')])
def OBVGraph(veri):

    from pandas_datareader import data as pdr
    import numpy as np

    hisse = veri+ '-USD'

    dfa = pdr.get_data_yahoo(hisse, start='2020-01-01', end='2021-02-07')
    df = dfa[len(dfa)-60:]
    df.reset_index(inplace=True)



    OBV = []
    OBV.append(0)
    for i in range(1,len(df['Close'])):
        if df['Close'][i]>df['Close'][i-1]:
            OBV.append(OBV[-1]+df['Volume'][i])
        elif df['Close'][i]<df['Close'][i-1]:
            OBV.append(OBV[-1]-df['Volume'][i])

        else:
            try:
                OBV.append[OBV[-1]]
            except:
                pass

    df['OBV'] = OBV
    df['OBV_EMA'] = df['OBV'].ewm(span=40).mean()
    import pandas as pd

    buy = []
    sell = []

    buy.append(np.nan)
    sell.append(np.nan)

    flag=1

    for i in range(1, len(df['Close'])):
        if df['OBV'][i]>df['OBV_EMA'][i] and flag !=1:
            buy.append(df['Close'][i])
            sell.append(np.nan)
            flag=1

        elif df['OBV'][i]<df['OBV_EMA'][i] and flag !=0:
            sell.append(df['Close'][i])
            buy.append(np.nan)
            flag=0
        else:
            buy.append(np.nan)
            sell.append(np.nan)


    df['Buy'] = buy
    df['Sell'] = sell


    layout = go.Layout(
        title=hisse,
        height=1000
    )

    trace0 = go.Scatter(x=df['Date'], y=df['Close'], mode='lines')
    trace1 = go.Scatter(x=df['Date'], y=df['Buy'], mode='markers')
    trace2 = go.Scatter(x=df['Date'], y=df['Sell'], mode='markers')

    fig = go.Figure([trace0,trace1,trace2], layout=layout)

    return fig

if __name__ == '__main__':
    app.run_server(debug=False)