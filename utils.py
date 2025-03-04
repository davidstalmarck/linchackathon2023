import pandas as pd
import numpy as np
import math


def moving_average(L: list[int], n):
    if len(L)>n:
        newL = L[-n:]
    else:
        newL= L
    return sum(newL[-n:]) / len(newL)


def computeRSI(data, window_size):
    diff = data.diff(1).dropna()        # find the difference between the current and previous value
    up_chg = 0 * diff
    down_chg = 0 * diff
    up_chg[diff > 0] = diff[ diff>0 ]
    down_chg[diff < 0] = diff[ diff < 0 ]
    up_chg_avg = up_chg.ewm(window_size).mean()
    down_chg_avg = down_chg.ewm(window_size).mean()
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    """
    data = pd.read_csv('AAPL.csv')
    data = data.set_index(pd.DatetimeIndex(data['Date'].values))
    rsi =
    """
    return rsi

