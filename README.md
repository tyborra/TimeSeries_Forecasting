# Time Series Forecasting
## Summary
This is based on three real, but unknown markets. The original code was done as part of a coding assesment. 

### <u>Part 1</u>
The primary challenge in this section is the very long forecast time. Most methods and algorithms will go astray when trying to predict this far out. The data contained several features that were largely irrelevant for the task, and some manipulation was required to get the data into a useful format. 

Analysis of the finalized data revealed a time series comprised of trend, season, and residual components. This, in addition to the long prediction window, led me to choose an ARIMA-based model, SARIMA(Seasonal ARIMA), specifically Statsmodels’ SARIMAX.
A grid search was conducted to find the optimal parameters of the SARIMAX model and then manual tuning was used to achieve the desired results.

### <u>Part 2</u>
Market1 and Market 2 were very similar datasets. The data required very little manipulation, which is typical for financial data. The only steps performed were to split the data (75/25) and normalize the data. I initially winsorized the data at the .01/.99 percentiles, but many of the entities were clipped for the majority of the validation data. This clipping did yield a significant improvement in MSE on the validation set but would have led to very poor forecasting results due to the data being constant for several periods. 

The data was converted into rolling 50-day windows with one prediction per window for training and evaluating the models. This method retains the time-series nature of the data and accurately mimics real-world time series forecasting.

The final predictions were made using an autoregressive method with the same rolling 50 period window making one prediction at a time. The prediction is added to the data to make the next prediction.
![autregressive.jfif](attachment:autregressive.jfif)


### <u>Market 1</u>
I chose to use an LSTM RNN implemented with TensorFlow for Market 1. The architecture consisted of four LSTM layers, each with dropout, a dense output layer, and rmsprop as the optimizer. The result was a mean MSE of ~0.08 on the validation data for all entities, reasonable for the scope of this project. The model appears as though it performed well on the 20 day forecast window.

### <u>Market 2</u>

In order to facilitate the reuse of code, I chose to stick with TensorFlow and use a GRU model for market 2. The model architecture consists of four GRU layers each with a dropout, and a dense output layer. I initially used SGD as the optimizer but the gradients exploded on the final model so I switched to rmsprop. I also experimented with a cyclical learning rate callback (https://arxiv.org/abs/1506.01186) and gradient clipping (https://papers.nips.cc/paper/2017/file/f2fc990265c712c49d51a18a32b39f0c-Paper.pdf). This model initially yielded a mean MSE of ~0.004 on the validation data for all entities, but performed poorly on the 20 day test window. The final model yielded a score of 0.1351 on the validation data, but appears to have performed well on the 20 day forecast window.

### <u>To Do</u>
- Modularity; the code needs to be organized into functions and classes 
- improve code efficiency, there is too much indexing
- annotate code
- tune the GRU model
