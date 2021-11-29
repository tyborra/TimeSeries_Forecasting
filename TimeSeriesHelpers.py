# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 10:35:24 2021
@author: Tyler Borras

-display_df(df, n_rows = 5)
-nan_check(df)
-print_unique(df)
-plot_decomposition(df)
-sarimax_grid_search(ts, max_iter, n_season, n_pdq = 3)
-perc_chng(df, periods = 1)
-create_time_dict(train_df, val_df = None , win_len = 50, create_val = True)
-fit_model_dict(model, train_dict, epochs, batch_size)
-validate_model_dict(model, val_dict, train_std, train_mean, plot_results = True)
-forecast_dict(model, train_dict, start_date, n_forecast = 20)

"""


import pandas as pd
import itertools
import warnings
import statsmodels.tsa.api as tsa
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, Latex
from statsmodels.tsa.stattools import adfuller
import numpy as np
from sklearn.metrics import mean_squared_error


def display_df(df, n_rows = 5):
    '''
    This displays the first n_rows rows of the df transposed. The option_context settings are to prevent data truncation
    '''   
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(df.head(n_rows).transpose()) 
        
        
def nan_check(df):
    '''
    Check df for NaN's, displays sum of NaN's by column
    '''    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(df.isna().sum())
        
        
def print_unique(df):
    '''
    print all unique columns of df
    '''
    for col in df:
        print('{0:<20}\t{1:4d}'.format(col, len(df[col].unique())))


def plot_decomposition(df):
    '''
    Plots seasonal decomposition from statsmodels seasonal_decompose.
    
    Parameters
    ----------
    df : univariate pandas df or df column
        
    Returns
    -------
    Plot

    '''
    df_components = tsa.seasonal_decompose(df, model='additive')
    ts = (df.to_frame('Original')
                  .assign(Trend=df_components.trend)
                  .assign(Seasonality=df_components.seasonal)
                  .assign(Residual=df_components.resid))
    
    with sns.axes_style('white'):
        ts.plot(subplots=True, figsize=(14, 8), title=['Original Series', 'Trend Component', 'Seasonal Component','Residuals'], 
                legend=False)
        plt.suptitle('Seasonal Decomposition', fontsize=14)
        sns.despine()
        plt.tight_layout()
        plt.subplots_adjust(top=.91);


def dickey_fuller(df, alpha = 0.05):
    '''
    

    Parameters
    ----------
    df :  univariate pandas df or df column
        
    alpha : alpha hor hypothesis test
        DESCRIPTION. The default is 0.05
        valid options are .01, .05, .10

    Returns
    -------
    Results

    '''
    cv_per = '{}%'.format(int(alpha*100))
    
    adf = adfuller(df)
    adf_stat = adf[0]
    crit_val = adf[4][cv_per]
    
    display(Latex(rf'$H_0=$ Data is not stationary''$\\\$'
                  rf'$H_A=$ Data is stationary (it has a time dependent structure)''$\\\$'
                  rf'$\alpha = {alpha}$''$\\\$'
                  rf'$p = {adf[1]:.4f}$''$\\\$'
                  rf'ADF Statistic $= {adf_stat:.2f}$''$\\\$'
                  rf'Critical Value $= {crit_val:.2f}$''$\\\$'
                 ))    
    
    if (adf[0] <= alpha) and (adf_stat < crit_val):
        display(Latex( rf'$p <= \alpha$ and {adf_stat:.2f} < {crit_val:.2f}, ' '$\\\$'),'Reject the null hypothesis, the data is stationary')
    
    elif (adf[0] > alpha) or (adf_stat > crit_val):
        display( Latex( rf'$p > \alpha$ or{adf_stat:.2f} > {crit_val:.2f}, ' '$\\\$'), 'Fail to reject the null hypothesis, the data is not stationary')
    
    else:
        display('Inconclusive')


def sarimax_grid_search(ts, max_iter, n_season, n_pdq = 3):
    '''
    Parameters
    ----------
    ts : univariate pandas df or df column
        
    max_iter : maximium iterations for each SARIMAX model
        
    n_season : seasonal value for SARIMAX
        
    n_pdq : number of p, d, and q to test. 
        DESCRIPTION. The default is 3.

    Returns
    -------
    min_aic : minimum aic
        
    min_bic : minimum bic
        
    '''
    
    p = range(0,n_pdq)
    d = range(0,n_pdq)
    q = range(0,n_pdq)
    
    pdq = list(itertools.product(p, d, q))
    pdqs = [(x[0], x[1], x[2], n_season) for x in list(itertools.product(p, d, q))]
    params = []
    warnings.filterwarnings('ignore')    #filter the warnings from Statsmodels
    
    for o in  pdq:
        for so in pdqs:            
            try:
                mod = tsa.statespace.SARIMAX(ts, order = o, seasonal_order = so).fit(maxiter=max_iter)
                params.append([o, so, mod.aic, mod.bic])
            except:
                continue
                
    param_df = pd.DataFrame(params, columns = ['pdq', 'pdqs', 'aic', 'bic'])
    min_aic = param_df[param_df['aic']==param_df['aic'].min()]
    min_bic = param_df[param_df['bic']==param_df['bic'].min()]
    return min_aic, min_bic


def perc_chng(df, periods = 1):
    '''   
    
    Parameters
    ----------
    df : pandas df
    periods: periods for pct_change
        DESCRIPTION. The default is 3.    
    Returns
    -------
    df : pandas df of percent change by column for periods number of periods
        
    '''
    for col in df.columns:
        df[col] = df[col].pct_change(periods = periods)
    df.drop(df.index[0], inplace=True)
    return df


def create_time_dict(train_df, val_df = None , win_len = 50, create_val = True):
    '''
    

    Parameters
    ----------
    train_df : train df
        
    val_df : validation df
        
    win_len : window length
        DESCRIPTION. The default is 50.

    Returns
    -------
    sanity check df
        
    train_dict : dictionary of train data
        
    val_dict : dictionary of validation data
        
    '''
    #split the data by entity and reshape the data
    if create_val:    
        entity_list = train_df.columns
        train_temp = {}
        val_temp = {}
    
        for ent in entity_list:
            train_temp[ent] = np.array(train_df[ent]).reshape(train_df.shape[0],1)
            val_temp[ent] = np.array(val_df[ent]).reshape(val_df.shape[0],1)
            
        
        #Create a dict of data for the model
        train_dict = {}
        val_dict = {}
    
        for ent in entity_list:
            train_dict[ent] = {}
            val_dict[ent] = {}
            X_train = []
            y_train = []         
    
            for i in range(win_len, len(train_df)):
                X_train.append(train_temp[ent][i-win_len:i,0])
                y_train.append(train_temp[ent][i,0])
    
            X_train, y_train = np.array(X_train), np.array(y_train)
            train_dict[ent]["X"] = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
            train_dict[ent]["y"] = y_train
    
            val_dict[ent] = {}
            X_val = []
            y_val = []   
    
            for i in range(win_len, len(val_df)):
                X_val.append(val_temp[ent][i-win_len:i,0])
                y_val.append(val_temp[ent][i,0])
    
            X_val, y_val = np.array(X_val), np.array(y_val)
            val_dict[ent]["X"] = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
            val_dict[ent]["y"] = y_val
    
        #sanity check
        arr_buff = []
        for ent in entity_list:
            buff = {}
            buff["X_train"] = train_dict[ent]["X"].shape
            buff["y_train"] = train_dict[ent]["y"].shape
            buff["X_val"] = val_dict[ent]["X"].shape
            buff["y_val"] = val_dict[ent]["y"].shape
            arr_buff.append(buff)
            train_shape = (X_train.shape[1],1)
        
        return pd.DataFrame(arr_buff, index=entity_list), train_dict, val_dict, train_shape
    
    else:
        entity_list = train_df.columns
        train_temp = {}


        for ent in entity_list:
            train_temp[ent] = np.array(train_df[ent]).reshape(train_df.shape[0],1)
        
        train_dict = {}

        for ent in entity_list:
            train_dict[ent] = {}
            X_train = []
            y_train = []
            
        
            for i in range(win_len, len(train_df)):
                X_train.append(train_temp[ent][i-win_len:i,0])
                y_train.append(train_temp[ent][i,0])
                
            X_train, y_train = np.array(X_train), np.array(y_train)
            train_dict[ent]["X"] = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
            train_dict[ent]["y"] = y_train
            
        #sanity check
        arr_buff = []
        for ent in entity_list:
            buff = {}
            buff["X_train"] = train_dict[ent]["X"].shape
            buff["y_train"] = train_dict[ent]["y"].shape
            arr_buff.append(buff)
            train_shape = (X_train.shape[1],1)
        
        return pd.DataFrame(arr_buff, index=entity_list), train_dict, train_shape
    
    

def fit_model_dict(model, train_dict, epochs, batch_size):
    '''
    

    Parameters
    ----------
    model : TF model
    train_dict : dict of traiiniing data
    epochs : num epochs
    batch_size : batch size

    Returns
    -------
    model : fitted model

    '''
    for ent in train_dict:
        print("Fitting ", ent)
        model.fit(train_dict[ent]["X"], train_dict[ent]["y"], 
                     epochs=epochs, 
                     batch_size=batch_size)
    return model


def validate_model_dict(model, val_dict, train_std, train_mean, plot_results = True):
    '''
    

    Parameters
    ----------
    model : Fitted TF model
    val_dict : validation dict
    train_std : standard deviiation of training data
    train_mean : mean of traiining data
    plot_results : Boolean to plot results
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    '''
    pred_result = {}
    overall_MSE = []

    for ent in val_dict:
        y_true = ((val_dict[ent]["y"].reshape(-1,1)) * train_std[ent]) + train_mean[ent]

        #Predict
        y_pred = ((model.predict(val_dict[ent]["X"])) * train_std[ent]) + train_mean[ent]        
        
        MSE = mean_squared_error(y_true, y_pred)
        overall_MSE.append(MSE)
        pred_result[ent] = {}
        pred_result[ent]['MSE']  = MSE
        pred_result[ent]["True"] = y_true
        pred_result[ent]["Pred"] = y_pred 
        
        if plot_results:            
            plt.figure(figsize=(14,6))
            plt.title("{}   with MSE{:10.4f}".format(ent,MSE))
            plt.plot(y_true)
            plt.plot(y_pred)
        
    print('Min MSE:    {0:.4f}\nMax MSE:    {1:.4f}\nMedian MSE: {2:.4f}\nMean MSE:   {3:.4f}'.format(np.min(overall_MSE), 
                                                                                              np.max(overall_MSE), 
                                                                                              np.median(overall_MSE), 
                                                                                              np.mean(overall_MSE)))


def forecast_dict(model, train_dict, start_date, n_forecast = 20):
    '''
    

    Parameters
    ----------
    model : Fitted TF model 
    train_dict : training diict
    start_date : start date of forecast
    n_forecast : numbeer of periods to forecast
        DESCRIPTION. The default is 20.

    Returns
    -------
    pred_df : df of predictions

    '''
    rng = pd.date_range(start_date, periods=n_forecast, freq='D').astype(str)
    pred_df = pd.DataFrame(columns = train_dict, index=rng)

    forecast_X = {}
    forecast_y = {}

    for ent in train_dict:    
        x = train_dict[ent]['X'][349][1:].reshape(-1, 49, 1)
        forecast_X[ent] = x
        forecast_y[ent] = train_dict[ent]['y'][-50:].reshape( 50, 1)


    for ent in train_dict:
        for i in range(n_forecast):     

            if(i == 0):
                print('Initializing x,y')
                y = forecast_y[ent][-1]
                x = forecast_X[ent][0:]
                X = np.append(x, y).reshape(-1,50,1)
            print('Predicting {}  forecast period {} '.format( ent, i+1))

            y_pred = (model.predict(X)) 

    #         LSTM_rnn.fit(X, y_new, 
    #                   epochs=1,    
    #                   batch_size=50,
    #                   #callbacks = [clr_triangular]
    #                  ) 

            y_new = y_pred[0][:].reshape(-1,1)
            X = np.append(X[0][1:], y_new).reshape(-1,50,1)    

        #write to df
        print('Writing to df')
        pred_df[ent] = X[0][-n_forecast:]  
    return pred_df


from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
#import numpy as np

class CyclicLR(Callback):
    """
    from https://github.com/bckenstler/CLR
    
    Copyright (c) 2017 Bradley Kenstler

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    # About
    This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())