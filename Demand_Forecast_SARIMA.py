# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:41:10 2020

@author: sattwik
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#%matplotlib inline
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import itertools
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


co_df = pd.read_csv('J:\MMM_Data\customer_orders_df.csv')		#enter the location of the customer_orders_df.csv file

co_df.rename(columns={'Requested Date': 'Requested_Date', 'Creation Date': 'Creation_Date', 'Shipped Date':'Shipped_Date'}, inplace=True)
co_df.rename(columns={'Order Number': 'Order_Number', 'Order Line': 'Order_Line', 'Requested Quantity':'Requested_Quantity', 'Shipped Quantity':'Shipped_Quantity'}, inplace=True)

co_df['Creation_Date'] =pd.to_datetime(co_df.Creation_Date, format='%d.%m.%Y %H:%M')
co_df['Requested_Date'] =pd.to_datetime(co_df.Requested_Date, format='%d.%m.%Y')
co_df['Shipped_Date'] =pd.to_datetime(co_df.Shipped_Date, format='%d.%m.%Y')

co_df['Delay']=(co_df['Shipped_Date']-co_df['Requested_Date']).dt.days
co_df["Deficit"]=co_df["Requested_Quantity"]-co_df["Shipped_Quantity"]

co_df_delay=co_df.loc[co_df["Delay"]<0]
drop_Req_Quant=co_df.index[co_df["Requested_Quantity"] <= 0].tolist()
co_df=co_df.drop(co_df.index[drop_Req_Quant])
co_df.sort_values(by=['Requested_Date'], inplace=True, ascending=True)

co_df["Indexdate"]=co_df["Requested_Date"]
co_df_indexed=co_df.set_index(["Indexdate"])


co_df_part = co_df.loc[co_df["Part"]=="8YE32821942352L"]

cnt = 0
demand_forecast=[]
for outlet in range(1,11):
    
    co_df_part_outlet = co_df_part.loc[co_df_part["Outlet"]==outlet]
    co_df_part_outlet["Indexdate"]=co_df_part_outlet["Requested_Date"]
    co_df_part_outlet = co_df_part_outlet.set_index(["Indexdate"])
    co_df_part_outlet = co_df_part_outlet.resample('W').sum()
    
    p = d = q = range(0,2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 52) for x in list(itertools.product(p, d, q))]
    
    best_aic = np.inf   
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(co_df_part_outlet['Requested_Quantity'],
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
    
                results=mod.fit()
                curr_aic=results.aic
                if curr_aic<=best_aic:
                    best_aic=curr_aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal
                    
            except:
                continue
        
    test = co_df_part_outlet['Requested_Quantity'][-6:]
    train = co_df_part_outlet['Requested_Quantity'][:-6]
    
    modplt = sm.tsa.statespace.SARIMAX(train,
                                    order=best_pdq,
                                    seasonal_order=best_seasonal_pdq,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    resultsplt = modplt.fit() 
    
    pred = resultsplt.predict(start = test.index[0], end = test.index[-1], dynamic = True)
    plt.figure(figsize = (20, 10))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.plot(pred.index, pred, label='SARIMA')
    plt.legend(loc='best');
    
    mod = sm.tsa.statespace.SARIMAX(co_df_part_outlet['Requested_Quantity'],
                                    order=best_pdq,
                                    seasonal_order=best_seasonal_pdq,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    
    f = results.forecast(8)		#forecasts demand for next eight weeks
    demand_forecast.append([])
    for i in range(8):
        demand_forecast[-1].append(f[i])
        
for i in range(10):
    for j in range(8):
        if demand_forecast[i][j]<=0:
            demand_forecast[i][j]=0

import csv
with open("J:\MMM_Data\Barkawi_Forecast_8.csv", "w") as f: 			#This is the output file. Change the file location as per preference
    writer = csv.writer(f)
    writer.writerows(demand_forecast)   
    