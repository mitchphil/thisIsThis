#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score, classification_report

import pickle
import gzip

nInst=100
currentPos = np.zeros(nInst)

def getMyPosition (prcSoFar):
    global currentPos
    (nins,nt) = prcSoFar.shape
    
    formattedDf = formatData(prcSoFar)
    daily_predict = formattedDf.copy()
    
    daily_predict['change_in_price'] = daily_predict.groupby('instrument')['price'].diff()
    daily_predict = calcRsi14(daily_predict)
    daily_predict = calcStochRsi14(daily_predict)
    daily_predict = calcStochOsc(daily_predict)
    daily_predict = calcMacd(daily_predict)
    daily_predict = calcProc(daily_predict)
    daily_predict = calcBoll(daily_predict)
    
    daily_predict_calc = daily_predict.loc[(daily_predict['day'] == (nt-1)), ['RSI_14','STOCH_RSI','k_percent','Price_Rate_Of_Change','MACD','MACD_EMA','BOLL']]
    
    with gzip.open('rfc_gzipped.pickle', 'rb') as f:
        rfc = pickle.load(f)
    with gzip.open('rfc_up_down_g.pickle', 'rb') as f:
        rfc_ud = pickle.load(f)
    with gzip.open('rfc_comm_g.pickle', 'rb') as f:
        rfc_comm = pickle.load(f)
    
    # NaN Handling
    
    daily_predict_calc.replace([np.inf, -np.inf], np.nan, inplace=True)
    daily_predict_calc['BOLL'] = daily_predict_calc['BOLL'].fillna(0)
    daily_predict_calc['k_percent'] = daily_predict_calc['k_percent'].fillna(50)
    daily_predict_calc['STOCH_RSI'] = daily_predict_calc['STOCH_RSI'].fillna(50)
    daily_predict_calc['RSI_14'] = daily_predict_calc['RSI_14'].fillna(50)
    
    rpos = rfc.predict(daily_predict_calc)
    nxt_day_ud = rfc_ud.predict(daily_predict_calc)
    comm = rfc_comm.predict(daily_predict_calc)
    
    dp = fixedPosSize(rpos, currentPos, daily_predict.loc[daily_predict['day'] == nt-1, ['price']].values.flatten(), daily_predict.loc[daily_predict['day'] == nt-1, ['change_in_price']].values.flatten(), nxt_day_ud, comm)
    currentPos = dp
    
    return currentPos

    
def formatData(dataset):
    
    (nins,nt) = dataset.shape
    
    df = pd.DataFrame(dataset.transpose())

    formatted_df = pd.DataFrame(columns=['instrument', 'day', 'price'])
    i = 0
    while i < nins:
        p = df[i]
        p = p.values.flatten()
        instr = np.full((nt,),i)
        d = np.linspace(0,nt-1, num=nt)

        stock_df = pd.DataFrame({'instrument':instr, 'day':d, 'price':p})
        formatted_df = formatted_df.append(stock_df, ignore_index=True)

        i+=1
    
    return formatted_df

def calcRsi14(df):
    
    n = 14

    up_df, down_df = df[['instrument','change_in_price']].copy(), df[['instrument','change_in_price']].copy()
    up_df.loc['change_in_price'] = up_df.loc[(up_df['change_in_price'] < 0), 'change_in_price'] = 0
    down_df.loc['change_in_price'] = down_df.loc[(down_df['change_in_price'] > 0), 'change_in_price'] = 0
    down_df['change_in_price'] = down_df['change_in_price'].abs()

    ewma_up = up_df.groupby('instrument')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())
    ewma_down = down_df.groupby('instrument')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())

    relative_strength = ewma_up / ewma_down

    relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

    df['down_days'] = down_df['change_in_price']
    df['up_days'] = up_df['change_in_price']
    df['RSI_14'] = relative_strength_index
    
    return df

def calcStochRsi14(df):
    
    n = 14

    low_14, high_14 = df[['instrument','RSI_14']].copy(), df[['instrument','RSI_14']].copy()

    low_14 = low_14.groupby('instrument')['RSI_14'].transform(lambda x: x.rolling(window = n).min())
    high_14 = high_14.groupby('instrument')['RSI_14'].transform(lambda x: x.rolling(window = n).max())

    STOCH_RSI = (df['RSI_14'] - low_14) / (high_14 - low_14)

    df['STOCH_RSI'] = STOCH_RSI

    return df

def calcStochOsc(df):
    
    n = 14

    low_14, high_14 = df[['instrument','price']].copy(), df[['instrument','price']].copy()

    low_14 = low_14.groupby('instrument')['price'].transform(lambda x: x.rolling(window = n).min())
    high_14 = high_14.groupby('instrument')['price'].transform(lambda x: x.rolling(window = n).max())

    k_percent = 100 * ((df['price'] - low_14) / (high_14 - low_14))

    df['low_14'] = low_14
    df['high_14'] = high_14
    df['k_percent'] = k_percent

    return df

def calcMacd(df):
    
    ema_26 = df.groupby('instrument')['price'].transform(lambda x: x.ewm(span = 26).mean())
    ema_12 = df.groupby('instrument')['price'].transform(lambda x: x.ewm(span = 12).mean())
    macd = ema_12 - ema_26

    ema_9_macd = macd.ewm(span = 9).mean()

    df['MACD'] = macd
    df['MACD_EMA'] = ema_9_macd

    return df

def calcProc(df):
    
    n = 9
    df['Price_Rate_Of_Change'] = df.groupby('instrument')['price'].transform(lambda x: x.pct_change(periods = n))
    return df

def calcBoll(df):
    
    n = 12
    sma12 = df.groupby('instrument')['price'].transform(lambda x: x.rolling(n).mean())
    std12 = df.groupby('instrument')['price'].transform(lambda x: x.rolling(n).std())
    
    upper = sma12 + 2*std12
    lower = sma12 - 2*std12
    
    BollingerVal = (2 * (df['price'] - sma12)) / (upper - lower)
    
    df['BOLL'] = BollingerVal
    
    return df
    
    
# Function calculates entry size - all buy and sell call have an inital $75 trade size, this amount allows for the ability to DCA all positions up to seven times in order to reduce risk of losing a max position to less than 2%.    
    
def fixedPosSize(pred_pos, current_pos, prc_today, change_in_price, ud, comm):
    
    pos = np.zeros(100)
    
    for t in range(0,100):
        
        pos_size = 75/prc_today[t] # Fixed size of position to $75 amount - to minimise risk - posistions can be DCAd up to 7 times
                                   # before pos limit is reached
        
        if current_pos[t] > 0:
            if pred_pos[t] < 0 and ud[t] < 0:
                pos[t] = 0
            else:
                if pred_pos[t] > 0 and change_in_price[t] < 0 and ud[t] > 0 and comm[t] > 0: # DCA the loss
                    pos[t] = 2*current_pos[t]
                elif pred_pos[t] == 0 and ud[t] < 0 and comm[t] < 0 and change_in_price[t] > 0: # Take some profits if momentum is sideways
                    pos[t] = 0.8*current_pos[t]
                else:
                    pos[t] = current_pos[t]
                    
        elif current_pos[t] < 0:
            if pred_pos[t] > 0 and ud[t] > 0:
                pos[t] = 0
            else:
                if pred_pos[t] < 0 and change_in_price[t] > 0 and ud[t] < 0 and comm[t] > 0: # DCA the loss
                    pos[t] = 2*current_pos[t]
                elif pred_pos[t] == 0 and ud[t] > 0 and comm[t] > 0 and change_in_price[t] < 0:  # Take some profits if momentum is sideways
                    pos[t] = 0.8*current_pos[t]
                else:
                    pos[t] = current_pos[t]
                    
        else:
            if pred_pos[t] < 0 and ud[t] < 0:
                pos[t] = -pos_size
            elif pred_pos[t] > 0 and ud[t] > 0:
                pos[t] = pos_size
            else:
                pos[t] = 0
                
    return pos





