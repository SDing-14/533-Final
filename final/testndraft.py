import pandas as pd
import numpy as np

predict_num = 100

ivv_price = pd.read_csv('ivv_price.csv', index_col='Date', parse_dates=True)
# ml = pd.read_csv('ml_ledger.csv', index_col='entry_dt', parse_dates=True)
ledger = pd.read_csv('original_ledger.csv', index_col='entry_dt', parse_dates=True)
# ledger = ledger[ledger['exit_dt'].notnull()]
# ledger['entry_dt'] = pd.to_datetime(ledger['entry_dt'])
# ledger.set_index('entry_dt', inplace=True)
ledger['return'] = ledger['return'].fillna(0)
ledger = ledger[-predict_num:-1]

ledger = ledger.merge(ivv_price[['Close Price']], left_index=True, right_index=True, how='left')
ledger = ledger[ledger['success'].notnull()]


# ml = ml.merge(ivv_price[['Close Price']], left_index=True, right_index=True, how='left')
# ml = ml.merge(ledger[['exit_dt']], left_index=True, right_index=True, how='left')
# ml = ml[ml['ML Actual'].notnull()]
# # ml['exit_dt'] = pd.to_datetime(ml['exit_dt'])
# # ml['ivv_return'] = ml['Close Price'].pct_change()
# # ml.to_csv('ml.csv')
# # print(ml[ml['ML Actual'] == -1])

ledger['IVV Return'] = 0
for i in range(len(ledger)):
     if ledger.iloc[i, 6] != 0:
         sellday = ledger.iloc[i, 4]
         closeprice = ledger.loc[sellday, 'Close Price']
         ledger.iloc[i, 10] = np.log(closeprice/ledger.iloc[i, -2])


# ml = ml[ml['ML Actual'] != 0]
ledger = ledger[ledger['success'] != 0]

print(ledger.tail(50))





    