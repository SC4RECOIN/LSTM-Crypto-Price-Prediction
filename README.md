# LSTM-Cryptonalysis
Predicting the price of Bitcoin using an lstrm-RNN. Technical analysis is applied to historical BTC data in attempt to extract price for automated trading. 

## Label
The price of Bitcoin tends to be very volatile and sporadic making it difficult to find underlying trends and predict price reversals. In order to smooth the historical price data without indroducing latency, a [Savitzky-Golay filter](https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.signal.savgol_filter.html) is applied. This purpose of this filter is to smooth the data without greatly distorting the signal. The first-order derivative is then taken to find the slope of the filtered data to indicate upwards and downwards movements about the zero axis. This can be seen in the following figure:    
     
![alt text](docs/label_snip.PNG)
    
## Features
* MACD histogram
* Stochastic RSI
* Volume
* Interpolation of price
    
![alt text](docs/ta_analysis.PNG)   

An approximation of the next price is performed using ridge regression from Scikit-learn. Through polynomial interpolation, the price can be treated as a continuous function and the next value in a series can be approximated. 

![alt text](docs/poly_interpolation.png)

