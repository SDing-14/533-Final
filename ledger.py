import pandas as pd
from datetime import datetime
import numpy as np
import os
import eikon as ek
import refinitiv.data as rd
from pandas_market_calendars import get_calendar
#####################################################
eikon_api = '81c9d3dad1c54648a2082fe7e09ce3888df0dfac'
ek.set_app_key(eikon_api)

# ek.set_app_key(os.getenv('EIKON_API'))
global raw_prc

def create_ledger(blotter):
    blotter = pd.DataFrame(blotter)
    id_list = blotter['trade_id'].unique()
    nyse = get_calendar('NYSE')
    ledger = pd.DataFrame(blotter)
    ledger['index'] = ledger.index
    ledger = ledger.set_index(['index', 'trade_id'])
    ledger_df = pd.DataFrame(columns=['Trade', 'asset', 'entry_dt', 'entry', 'exit_dt', 'exit'
                                      , 'success', 'n', 'return'])
    for id in id_list:
        skip = False
        subdata = ledger.iloc[ledger.index.get_level_values('trade_id') == id]

        for index, row in subdata.iterrows():
            if row['status'] == 'LIVE':
                index_list = subdata.index.get_level_values('trade_id').tolist()

                new_df = pd.DataFrame(columns=['Trade', 'asset', 'entry_dt', 'exit_dt', 'exit',
                                            'success', 'n', 'return'])
                new_df.loc[0, 'Trade'] = index_list[0]
                new_df.loc[0, 'asset'] = subdata.loc[subdata.index[0], 'asset']
                new_df.loc[0, 'entry_dt'] = subdata.loc[subdata.index[0], 'date']
                new_df.loc[0, 'entry'] = subdata.loc[subdata.index[0], 'price']
                ledger_df = pd.concat([ledger_df, new_df])
                skip = True
                break

        if skip:
            continue

        index_list = subdata.index.get_level_values('trade_id').tolist()

        if len(subdata) == 4:
            new_df = pd.DataFrame(columns=['Trade', 'asset', 'entry_dt', 'exit_dt', 'exit',
                                        'success', 'n', 'return'])

            new_df.loc[0, 'success'] = 1
            new_df.loc[0, 'Trade'] = index_list[0]
            new_df.loc[0, 'asset'] = subdata.loc[subdata.index[0], 'asset']
            new_df.loc[0, 'entry_dt'] = subdata.loc[subdata.index[0], 'date']
            new_df.loc[0, 'entry'] = subdata.loc[subdata.index[0], 'price']
            new_df.loc[0, 'exit'] = subdata.loc[subdata.index[2], 'price']
            new_df.loc[0, 'exit_dt'] = subdata.loc[subdata.index[3], 'date']
            new_df.loc[0, 'return'] = np.log(subdata.loc[subdata.index[3], 'price']) - np.log(
                subdata.loc[subdata.index[0], 'price'])

            entry_date = subdata.loc[subdata.index[0], 'date']
            exit_date = subdata.loc[subdata.index[3], 'date']
            schedule = nyse.schedule(start_date=entry_date, end_date=exit_date)
            trading_days = schedule[schedule['market_open'].dt.dayofweek < 5].index.tolist()
            trading_days = len(trading_days)
            new_df.loc[0, 'n'] = trading_days

            ledger_df = pd.concat([ledger_df, new_df])

        elif len(subdata) == 2:
            new_df = pd.DataFrame(columns=['Trade', 'asset', 'entry_dt', 'exit_dt', 'exit',
                                        'success', 'n', 'return'])
            new_df.loc[0, 'success'] = 0
            new_df.loc[0, 'Trade'] = index_list[0]
            new_df.loc[0, 'asset'] = subdata.loc[subdata.index[0], 'asset']
            new_df.loc[0, 'entry_dt'] = subdata.loc[subdata.index[0], 'date']
            new_df.loc[0, 'entry'] = subdata.loc[subdata.index[0], 'price']
            ledger_df = pd.concat([ledger_df, new_df])


        elif len(subdata) == 5:
            new_df = pd.DataFrame(columns=['Trade', 'asset', 'entry_dt', 'exit_dt', 'exit',
                                        'success', 'n', 'return'])
            new_df.loc[0, 'success'] = -1
            new_df.loc[0, 'Trade'] = index_list[0]
            new_df.loc[0, 'asset'] = subdata.loc[subdata.index[0], 'asset']
            new_df.loc[0, 'entry_dt'] = subdata.loc[subdata.index[0], 'date']
            new_df.loc[0, 'entry'] = subdata.loc[subdata.index[0], 'price']
            new_df.loc[0, 'exit'] = subdata.loc[subdata.index[4], 'price']
            new_df.loc[0, 'exit_dt'] = subdata.loc[subdata.index[4], 'date']
            new_df.loc[0, 'return'] = np.log(subdata.loc[subdata.index[4], 'price']) - np.log(subdata.loc[subdata.index[0], 'price'])

            entry_date = subdata.loc[subdata.index[0], 'date']
            exit_date = subdata.loc[subdata.index[4], 'date']
            schedule = nyse.schedule(start_date=entry_date, end_date=exit_date)
            trading_days = schedule[schedule['market_open'].dt.dayofweek < 5].index.tolist()
            trading_days = len(trading_days)
            new_df.loc[0, 'n'] = trading_days
            ledger_df = pd.concat([ledger_df, new_df])

    ledger_df.to_csv('original_ledger.csv')
    return ledger_df

