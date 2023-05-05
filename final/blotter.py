import pandas as pd
from datetime import datetime
import numpy as np
import os
import eikon as ek
import refinitiv.data as rd

#####################################################
eikon_api = '81c9d3dad1c54648a2082fe7e09ce3888df0dfac'
ek.set_app_key(eikon_api)


# ek.set_app_key(os.getenv('EIKON_API'))
global raw_prc
def query_data(start_date_str, end_date_str, instruments):
    global raw_prc
    
    prc, prc_err = ek.get_data(
        instruments=[instruments],
        fields=[
            'TR.OPENPRICE(Adjusted=0)',
            'TR.HIGHPRICE(Adjusted=0)',
            'TR.LOWPRICE(Adjusted=0)',
            'TR.CLOSEPRICE(Adjusted=0)',
            'TR.PriceCloseDate'
        ],
        parameters={
            'SDate': start_date_str,
            'EDate': end_date_str,
            'Frq': 'D'
        }
    )

    prc['Date'] = pd.to_datetime(prc['Date']).dt.date
    prc.drop(columns='Instrument', inplace=True)
    raw_prc = prc
    return


def make_submitted_entry_orders(prc, alpha1):
    # submitted entry orders
    submitted_entry_orders = pd.DataFrame({
        "trade_id": range(1, prc.shape[0]),
        "date": list(pd.to_datetime(prc["Date"].iloc[1:]).dt.date),
        "asset": "IVV",
        "trip": 'ENTER',
        "action": "BUY",
        "type": "LMT",
        "price": round(
            prc['Close Price'].iloc[:-1] * (1 + alpha1),
            2
        ),
        'status': 'SUBMITTED'
    })
    return submitted_entry_orders


def make_cancelled_entry_orders(submitted_entry_orders, prc, n1):
    with np.errstate(invalid='ignore'):
        cancelled_entry_orders = submitted_entry_orders[
            np.greater(
                prc['Low Price'].iloc[1:][::-1].rolling(n1).min()[::-1].to_numpy(),
                submitted_entry_orders['price'].to_numpy()
            )
        ].copy()
    cancelled_entry_orders.reset_index(drop=True, inplace=True)
    cancelled_entry_orders['status'] = 'CANCELLED'
    cancelled_entry_orders['date'] = pd.DataFrame(
        {'cancel_date': submitted_entry_orders['date'].iloc[(n1 - 1):].to_numpy()},
        index=submitted_entry_orders['date'].iloc[:(1 - n1)].to_numpy()
    ).loc[cancelled_entry_orders['date']]['cancel_date'].to_list()

    return cancelled_entry_orders


def make_filled_live_entry_orders(submitted_entry_orders, cancelled_entry_orders, prc, next_business_day, n1, alpha1):
    filled_entry_orders = submitted_entry_orders[
        submitted_entry_orders['trade_id'].isin(
            list(
                set(submitted_entry_orders['trade_id']) - set(
                    cancelled_entry_orders['trade_id']
                )
            )
        )
    ].copy()

    filled_entry_orders.reset_index(drop=True, inplace=True)
    filled_entry_orders['status'] = 'FILLED'

    for i in range(0, len(filled_entry_orders)):

        idx1 = np.flatnonzero(
            prc['Date'] == filled_entry_orders['date'].iloc[i]
        )[0]

        ivv_slice = prc.iloc[idx1:(idx1 + n1)]['Low Price']

        fill_inds = ivv_slice <= filled_entry_orders['price'].iloc[i]

        if (len(fill_inds) < n1) & (not any(fill_inds)):
            filled_entry_orders.at[i, 'status'] = 'LIVE'
        else:
            filled_entry_orders.at[i, 'date'] = prc['Date'].iloc[
                fill_inds.idxmax()
            ]

    live_entry_orders = pd.DataFrame({
        "trade_id": prc.shape[0],
        "date": pd.to_datetime(next_business_day).date(),
        "asset": "IVV",
        "trip": 'ENTER',
        "action": "BUY",
        "type": "LMT",
        "price": round(prc['Close Price'].iloc[-1] * (1 + alpha1), 2),
        'status': 'LIVE'
    },
        index=[0]
    )

    if any(filled_entry_orders['status'] == 'LIVE'):
        live_entry_orders = pd.concat([
            filled_entry_orders[filled_entry_orders['status'] == 'LIVE'],
            live_entry_orders
        ])
        live_entry_orders['date'] = pd.to_datetime(next_business_day).date()

    filled_entry_orders = filled_entry_orders[
        filled_entry_orders['status'] == 'FILLED'
        ]

    return filled_entry_orders, live_entry_orders


def make_entry_orders(submitted_entry_orders, cancelled_entry_orders, filled_entry_orders, live_entry_orders):
    entry_orders = pd.concat(
        [
            submitted_entry_orders,
            cancelled_entry_orders,
            filled_entry_orders,
            live_entry_orders
        ]
    ).sort_values(["date", 'trade_id'])

    return entry_orders


def make_submitted_exit_orders(filled_entry_orders, alpha2):
    submitted_exit_orders = filled_entry_orders.reset_index(drop=True)
    submitted_exit_orders['status'] = 'SUBMITTED'
    submitted_exit_orders['trip'] = 'EXIT'
    submitted_exit_orders['action'] = 'SELL'
    submitted_exit_orders['price'] = submitted_exit_orders['price'] * (1 + alpha2)

    return submitted_exit_orders


def make_market_exit_orders(cancelled_exit_orders_raw):
    market_exit_orders = pd.DataFrame({
        "trade_id": cancelled_exit_orders_raw['trade_id'],
        "date": cancelled_exit_orders_raw['date'],
        "asset": "IVV",
        "trip": 'EXIT',
        "action": "SELL",
        "type": "MKT",
        "price": cancelled_exit_orders_raw['n2_close_price'],
        'status': 'FILLED'
    })
    return market_exit_orders


def make_cancelled_exit_orders(prc, submitted_exit_orders, n2):
    #  比较 是否下单的时候在当天的high 前或后， open 和 close是否为high or low
    prc['rolling high'] = prc['High Price'][::-1].rolling(n2).max()[::-1].to_numpy()
    prc['n2_date_shift'] = prc['Date'][::-1].shift(n2 - 1)[::-1]
    prc['n2_close_price'] = prc['Close Price'][::-1].shift(n2 - 1)[::-1]
    prc = prc[['Date', 'rolling high', 'Close Price', 'n2_date_shift', 'n2_close_price']]

    cancelled_exit_orders_raw = pd.merge(submitted_exit_orders, prc, how='left', left_on='date', right_on='Date')
    cancelled_exit_orders_raw = cancelled_exit_orders_raw[
        cancelled_exit_orders_raw['price'] > cancelled_exit_orders_raw['rolling high']]
    cancelled_exit_orders_raw['status'] = 'CANCELLED'
    cancelled_exit_orders_raw['date'] = cancelled_exit_orders_raw['n2_date_shift']

    cancelled_exit_orders = cancelled_exit_orders_raw.drop(['rolling high',
                                                            'Date',
                                                            'Close Price',
                                                            'n2_date_shift',
                                                            'n2_close_price'], axis=1)

    return cancelled_exit_orders, cancelled_exit_orders_raw


def make_filled_live_exit_orders(submitted_exit_orders, cancelled_exit_orders, prc, n2, next_business_day):
    # filled exit orders & live exit orders

    filled_exit_orders = submitted_exit_orders[
        submitted_exit_orders['trade_id'].isin(
            list(
                set(submitted_exit_orders['trade_id']) - set(
                    cancelled_exit_orders['trade_id']
                )
            )
        )
    ].copy()

    filled_exit_orders.reset_index(drop=True, inplace=True)
    filled_exit_orders['status'] = 'FILLED'

    for i in range(0, len(filled_exit_orders)):

        idx1 = np.flatnonzero(
            prc['Date'] == filled_exit_orders['date'].iloc[i]
        )[0]

        ivv_slice = prc.iloc[idx1:(idx1 + n2)]['High Price']

        fill_inds = ivv_slice >= filled_exit_orders['price'].iloc[i]

        if (len(fill_inds) < n2) & (not any(fill_inds)):
            filled_exit_orders.at[i, 'status'] = 'LIVE'
        else:
            filled_exit_orders.at[i, 'date'] = prc['Date'].iloc[
                fill_inds.idxmax()
            ]

    live_exit_orders = filled_exit_orders[filled_exit_orders['status'] == 'LIVE']
    live_exit_orders.reset_index(drop=True, inplace=True)
    live_exit_orders['date'] = pd.to_datetime(next_business_day).date()

    filled_exit_orders = filled_exit_orders[~(filled_exit_orders['status'] == 'LIVE')]
    filled_exit_orders.reset_index(drop=True, inplace=True)

    return filled_exit_orders, live_exit_orders


def make_exit_orders(submitted_exit_orders, cancelled_exit_orders, filled_exit_orders, live_exit_orders,
                     market_exit_orders):
    exit_orders = pd.concat(
        [
            submitted_exit_orders,
            cancelled_exit_orders,
            filled_exit_orders,
            live_exit_orders,
            market_exit_orders
        ]
    ).sort_values(["date", 'trade_id'])
    return exit_orders


# def make_blotter(start_date_str, end_date_str, instruments, alpha1, n1, alpha2, n2):
def make_blotter(alpha1, n1, alpha2, n2):
    global raw_prc
    prc = raw_prc
    # prc = query_data(start_date_str, end_date_str, instruments)
    ##### Get the next business day from Refinitiv!!!!!!!
    ######################################################
    rd.open_session()
    next_business_day = rd.dates_and_calendars.add_periods(
        start_date=prc['Date'].iloc[-1].strftime("%Y-%m-%d"),
        period="1D",
        calendars=["USA"],
        date_moving_convention="NextBusinessDay",
    )
    rd.close_session()
    ######################################################

    submitted_entry_orders = make_submitted_entry_orders(prc, alpha1)
    cancelled_entry_orders = make_cancelled_entry_orders(submitted_entry_orders, prc, n1)
    filled_entry_orders, live_entry_orders = make_filled_live_entry_orders(submitted_entry_orders,
                                                                           cancelled_entry_orders, prc,
                                                                           next_business_day, n1, alpha1)
    entry_orders = make_entry_orders(submitted_entry_orders, cancelled_entry_orders, filled_entry_orders,
                                     live_entry_orders)

    # EXIT Order
    submitted_exit_orders = make_submitted_exit_orders(filled_entry_orders, alpha2)
    cancelled_exit_orders, cancelled_exit_orders_raw = make_cancelled_exit_orders(prc, submitted_exit_orders, n2)
    market_exit_orders = make_market_exit_orders(cancelled_exit_orders_raw)
    filled_exit_orders, live_exit_orders = make_filled_live_exit_orders(submitted_exit_orders, cancelled_exit_orders,
                                                                        prc, n2, next_business_day)

    exit_orders = make_exit_orders(submitted_exit_orders, cancelled_exit_orders, filled_exit_orders, live_exit_orders,
                                   market_exit_orders)

    blotter = pd.concat([entry_orders, exit_orders]).sort_values(["date", 'trade_id'])
    blotter.reset_index(drop=True, inplace=True)

    return blotter.sort_values(['trade_id',"date"])
