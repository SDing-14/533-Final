import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier


global window
global predict_num
global scaled_data
global return_history
global returns
window = 10
predict_num = 100


def train_and_result(ledger):
    global window
    global predict_num
    global scaled_data
    global return_history
    global returns

    data = pd.read_csv('data.csv', index_col='Dates', parse_dates=True)
    # ledger = pd.read_csv('ledger.csv', index_col='entry_dt', parse_dates=True)
    ledger = pd.DataFrame(ledger)
    ledger['entry_dt'] = pd.to_datetime(ledger['entry_dt'])
    ledger.set_index('entry_dt', inplace=True)

    df = pd.merge(ledger, data, how='left', left_index=True, right_index=True)

    return_history = []

    df = df[-200:]

    df = df.drop(['asset', 'Trade', 'exit_dt', 'exit',
                 'return', 'n', 'entry'], axis=1)
    success = df.pop('success')
    scaler = preprocessing.StandardScaler().fit(df)
    scaled_data = scaler.transform(df)
    scaled_data = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
    scaled_data['success'] = success

    df = df.drop(['IVV US Equity', 'FDTRFTRL Index',
                  'XAU Curncy', 'ECRPUS 1Y Index', 'JPYUSD Curncy', 'JPYUSD Curncy', 'USCRWTIC Index'], axis=1)

    scaled_data = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
    for col in scaled_data.columns:
        if col != 'success':
            scaled_data[col] = scaled_data[col].pct_change()

    scaled_data = scaled_data.dropna()
    scaled_data['IVV AU Equity']['2022-12-07'] = 0
    scaled_data['success'] = success

    scaled_data.dropna(inplace=True)
    scaled_data['success'] = scaled_data['success'].replace([-1, 0], 0)

    for i in range(0, predict_num):
        data = scaled_data[-1*window-i-2:-i-2]
        X = data.drop(['success'], axis=1)
        y = data['success']
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        result = clf.predict(scaled_data.drop(['success'], axis=1)[-i-2:-i-1])
        return_history.append(result[0])

    return_history = pd.DataFrame(
        return_history, index=scaled_data.index[-predict_num:])
    returns = ledger[['success']]
    returns = pd.merge(returns, return_history, how='left',
                       left_index=True, right_index=True)[-(predict_num+window):]
    returns.columns = ['Original Success', 'ML Decision']
    returns.reset_index(inplace=True)
    returns['entry_dt'] = pd.to_datetime(returns['entry_dt']).dt.date
    returns['ML Actual'] = returns['ML Decision']
    returns.loc[(returns['Original Success'] == 0) & (
        returns['ML Decision'] == 1), 'ML Actual'] = 0
    returns.loc[(returns['Original Success'] == -1) &
                (returns['ML Decision'] == 1), 'ML Actual'] = -1

    return returns.fillna('Training Window/Live Order')


##################################### Plots ##########################################

def ml_return(ledger):
    global window
    global predict_num
    # global returns

    returns.set_index('entry_dt', inplace=True)
    ledger = pd.DataFrame(ledger)
    ledger['entry_dt'] = pd.to_datetime(ledger['entry_dt'])
    ledger.set_index('entry_dt', inplace=True)
    ledger['return'] = ledger['return'].fillna(0)
    ml_return = returns[['ML Actual']]
    ml_return = ml_return.merge(
        ledger[['return']], left_index=True, right_index=True, how='left')
    ml_return = ml_return.rename(columns={'return': 'ledger return'})
    ml_return.loc[(ml_return['ML Actual'] == 0), 'ledger return'] = 0

    ml_return = ml_return[-predict_num:-1]

    ivv_price = pd.read_csv('ivv_price.csv', index_col='Date', parse_dates=True)
    ledger = ledger[ledger['exit_dt'].notnull()]
    ml_return = ml_return.merge(ivv_price[['Close Price']], left_index=True, right_index=True, how='left')
    ml_return = ml_return.merge(ledger[['exit_dt']], left_index=True, right_index=True, how='left')
    ml_return = ml_return[ml_return['ML Actual'].notnull()]

    ml_return['IVV Return'] = 0
    for i in range(len(ml_return)):
        if ml_return.iloc[i, 0] != 0:
            sellday = ml_return.iloc[i, 3]
            closeprice = ml_return.loc[sellday, 'Close Price']
            ml_return.iloc[i, 4] = np.log(closeprice/ml_return.iloc[i, 2])

    ml_return = ml_return[ml_return['ML Actual'] != 0]



    x = ml_return['IVV Return']
    y = ml_return['ledger return']

    # create some sample data
    # add a constant to the independent variable
    x1 = sm.add_constant(x)
    # create a linear regression model and fit it to the data
    model = sm.OLS(y, x1)
    results = model.fit()

    # get the values of alpha and beta
    alpha = results.params[0]
    beta = results.params[1]

    return x, y, alpha, beta


def ledger_return(ledger):
    global window
    global predict_num
    # ledger = pd.read_csv('ledger.csv', index_col='entry_dt', parse_dates=True)
    ledger = pd.DataFrame(ledger)
    ledger['entry_dt'] = pd.to_datetime(ledger['entry_dt'])
    ledger.set_index('entry_dt', inplace=True)
    ledger['return'] = ledger['return'].fillna(0)
    ledger = ledger[-predict_num:-1]


    ivv_price = pd.read_csv('ivv_price.csv', index_col='Date', parse_dates=True)
    ledger = ledger.merge(ivv_price[['Close Price']], left_index=True, right_index=True, how='left')

    ledger = ledger[ledger['success'].notnull()]


    ledger['IVV Return'] = 0
    for i in range(len(ledger)):
        if ledger.iloc[i, 5] != 0:
            sellday = ledger.iloc[i, 3]
            closeprice = ledger.loc[sellday, 'Close Price']
            ledger.iloc[i, -1] = np.log(closeprice/ledger.iloc[i, -2])

    ledger = ledger[ledger['success'] != 0]

    # Create some sample data
    x = ledger['IVV Return']
    y = ledger['return']

    ###### Alpha & Beta #######
    x1 = sm.add_constant(x)

    # create a linear regression model and fit it to the data
    model = sm.OLS(y, x1)
    results = model.fit()

    # get the values of alpha and beta
    alpha = results.params[0]
    beta = results.params[1]

    return x, y, alpha, beta
