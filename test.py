# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:40:29 2022

@author: oadiguzel
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import os

def get_data():
    return pd.read_csv("data.csv").set_index("date")

def zscore(df):
    for col in df.columns:
        col_zscore = col + '_zscore'
        df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)
    return df

def regression(df,Y,X):
    if type(Y) == list:
        Y = Y[0]
    str1 = " + ".join(str(i) for i in X)
    model = sm.ols(formula=f"{Y} ~ {str1}", data=df).fit()
    return model

def syntax(df, Yall, X, reg):
    beta = np.matrix(reg.params[1:])
    value = np.matrix(df[X])
    syn = np.array(value*beta.T + reg.params[0]).ravel()
    syn =  pd.DataFrame(syn,columns=["syntax"], index=df.index)
    syn = pd.merge(syn,df[Yall],left_index=True,right_index=True)
    return syn

def plot(df,size,smooting, smooting_period=21):
    df = df.sort_index(ascending=True)
    if smooting == True:
        df = df.rolling(smooting_period).mean().dropna()
    if size == "all":
        size = len(df)
    fig = df.tail(size).plot.line()
    fig.figure.savefig('demo.pdf')
    os.startfile("demo.pdf")
    return
    
    
if __name__ == "__main__":
    df = get_data()
    df_z = zscore(df)
    
    Yall = ["DJI","SPX","NDX","IXIC"]
    Y = input("dependent value  : ")
    Y = eval(Y)
    #Y = "DJI"
    X = input("independent values list; 'DXY','VIX','VXN','US10YT'  : ")
    X = list(eval(X))
    #X = ["DXY","VIX","VXN","US10YT"]
    reg = regression(df_z,Y,X)
    print(reg.summary())
    
    
    print("*"*50)
    print("R^2          : ", reg.rsquared)
    print("R^2_adj      : ", reg.rsquared_adj)
    print("*"*50)
    print("pvalues : ", "\n", "-"*10, "\n")
    print(reg.pvalues)
    print("*"*50)
    print("coefficients : ", "\n", "-"*15, "\n")
    print(reg.params)
    print("*"*50)
    
    is_all = input("draw Y/Yall : ")
    
    syn = syntax(df_z, eval(is_all), X, reg)
    a = int(input("graphic lenght : "))
    b = input("use smooting; True/False : ")
    c = input("smooting period : ")
    plot(syn,1000,True,21)
