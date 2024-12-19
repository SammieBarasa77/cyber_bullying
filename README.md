# Project: Amazon Stock Data Analysis
![Logo](https://github.com/SammieBarasa77/stock_analysis/blob/main/assets/images/stock_image.png)

## Table of Contents

   - [Introduction](#introduction)
   - [Key Aspects of the Stock Analysis](#key-aspects-of-the-stock-analysis)
   - [The Analysis](#the-analysis)
   - [Importing the Necessary Datasets](#importing-the-necessary-datasets)
   - [Feature Engineering](#feature-engineering)
   - [Exploratory Data Analysis](#exploratory-data-analysis)
     - [Correlation Between Different Features](#correlation-between-different-features)
     - [Decomposition of the Time Series](#decomposition-of-the-time-series)
     - [The ADF Statistic](#the-adf-statistic)
     - [Daily Returns Distribution](#daily-returns-distribution)
     - [Transforming the Data](#transforming-the-data)
   - [Train-Test Split](#train-test-split)
     - [Purpose of Train-Test Split](#purpose-of-train-test-split)
     - [Preparing the Training and Testing Series](#preparing-the-training-and-testing-series)
   - [Plotting the Differenced Data](#plotting-the-differenced-data)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
     - [Importance of Hyperparameter Tuning](#importance-of-hyperparameter-tuning)
   - [Regression with Random Forest](#regression-with-random-forest)
     - [Training, Predictions, and Model Evaluation](#training-predictions-and-model-evaluation)
     - [Plotting Actual vs Predicted Trends](#plotting-actual-vs-predicted-trends)
     - [Performance Evaluation in Percentage](#performance-evaluation-in-percentage)
  - [Final Verdict](#final-verdict)
  - [Presenting Findings](#presenting-findings)

## Introduction

Stock analysis is a critical process for understanding the dynamics of financial markets and making informed investment decisions. It involves evaluating historical price trends, trading volumes, and key financial indicators to assess a stock's performance and forecast future movements.

*By leveraging data-driven insights, stock analysis empowers investors to uncover patterns, evaluate market conditions, and identify potential opportunities or risks.*

Key aspects of the stock analysis include the following;
Historical Trend Analysis: Observing stock price movements over time to identify long-term trends, seasonality, or anomalies.

Volatility and Risk Assessment: Measuring the stock's price fluctuations and understanding market behavior during periods of uncertainty.

Fundamental Analysis: Examining company-specific metrics such as earnings, revenue growth, and valuation ratios.

Technical Analysis: Using chart patterns and indicators like moving averages, RSI, and MACD to predict price movements.

Sentiment Analysis: Gauging public and investor sentiment, which often impacts stock prices.

**The Analysis**

Anazon being one of the largest tech comapnies in globally, its stock performance often reflects broader market trends while also serving as a bellwether for the e-commerce, cloud computing, and technology sectors.

Analyzing Amazon's stock data offers insights into the following;

*How investors react to news, earnings reports, or global economic shifts.*

*Growth trajectory such as streaming, logistics, and AI are all part of its stock performance.*

**Volatility Trends**

In this project, we look at the Amazon stock data to uncover the following:

*Historic Price Trends: Understanding how Amazon's stock has performed.*

*Volume Analysis: Studying trading volumes to assess periods of heightened activity.*

*Volatility and Moving Averages: Evaluating Amazon's price fluctuations and applying technical indicators to identify support and resistance levels.*

*Seasonality and Key Events: Analyzing seasonal trends and the impact of significant announcements on stock price movements.*

*Forecasting Stock Performance: Using machine learning models like ARIMA to predict Amazon's future stock prices.*

**Importing the necessary datasets**

*The 'yfinance'library provides the stock market data necessary for the analysis. With it, you can download Amazon's historical stock data and key metrics to explore trends and build forecasts.*

```python
import pandas as pd 
import numpy as np
import yfinance as yf
```
*The yf.download() function retrieves historical stock data for Amazon from Yahoo Finance.*

*The data includes key metrics like Open, High, Low, Close, Adj Close, and Volume for each trading day.*

```python
# Fetchning the Amazon data using the Yahoo Finance API 

# Fetch AMZN stock data
ticker = "AMZN"
data = yf.download(ticker, start="2010-01-01", end="2024-12-31")
```

Output
![first_data prevw](https://github.com/SammieBarasa77/stock_analysis/blob/main/assets/images/first_data_preview.png)

```python
data
```
![sec_data_prvw](https://github.com/SammieBarasa77/stock_analysis/blob/main/assets/images/second_preview.png)

**Resetting Index**

Converting the Date index back to a column ensures that the Date field is explicitly accessible for analysis, plotting, or filtering operations.

**Handling Missing Values**

Dropping rows with missing values (data.dropna()) ensures data consistency and accuracy, which is crucial for reliable analysis and modeling.

**Converting Date to Datetime**

Ensuring the Date column is in datetime format facilitates efficient date-based filtering, aggregation (e.g., monthly or yearly), and plotting.

**Preparing for Scaling**

Using MinMaxScaler later requires clean and formatted numerical data, which this preprocessing step helps achieve.
```python
# Data Preprocessing 
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Reset index to make Date a column
data.reset_index(inplace=True)

# Handle missing values (fill or drop)
data = data.dropna()

# Convert Date to datetime
data['Date'] = pd.to_datetime(data['Date'])
```

## Feature Engineering

*Adding Moving Averages*

20-day Moving Average (MA20): Provides short-term trend insights, helping identify recent price movements and momentum.

50-day Moving Average (MA50): Captures medium-term trends, offering a smoother representation of price fluctuations over a longer period.

*Trend Identification*

These moving averages help detect potential buy or sell signals (e.g., when the MA20 crosses above or below the MA50) and assist in understanding the stock's overall direction.

*Smoothing Volatility*

Moving averages reduce daily price noise, making it easier to interpret patterns and trends in Procter & Gamble's stock performance.

```python
# Feature engineering

# Add moving averages 
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()
```

Adding RSI (Relative Strength Index)

RSI is a momentum oscillator that helps identify overbought or oversold conditions in the stock. It ranges from 0 to 100 and indicates potential reversal points when above 70 (overbought) or below 30 (oversold). This is useful for identifying buying or selling opportunities based on market sentiment.

Normalizing Features

MinMaxScaler transforms the Close and Volume values to a range between 0 and 1, making them suitable for machine learning models. This ensures that no single feature dominates due to differences in scale, improving model performance.

Handling NaN Values

Dropping rows with NaN values ensures that only complete and reliable data is used for further analysis, preventing errors or skewed results in subsequent steps.

```python
# Add RSI (Relative Strength Index)
delta = data['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Normalize features for machine learning
scaler = MinMaxScaler()
data[['Close_norm', 'Volume_norm']] = scaler.fit_transform(data[['Close', 'Volume']])

# Drop rows with NaN (resulting from rolling calculations)
data = data.dropna()

# Check the updated dataset
print(data.head())
```
Output
![RSI](https://github.com/SammieBarasa77/stock_analysis/blob/main/assets/images/RSI.png)

## Exploratory Data Analysis

Visualizing Stock Price Trends

The line plot provides a clear visual representation of how Procter & Gamble's stock has evolved over time. By plotting the closing prices, you can easily identify long-term trends, periods of stability, and significant fluctuations.

Identifying Patterns

The plot allows for the detection of patterns like upward or downward trends, seasonal fluctuations, and potential anomalies, which could signal important market events or company milestones.

Stock Trends for the AMAZON stock data
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Line plot for stock prices
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
plt.title("AMZN Stock Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()
```
Output
![Trends](https://github.com/SammieBarasa77/stock_analysis/blob/main/assets/images/trends.png)

### Correlation between Different Features
```python
# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data[['Close', 'Volume', 'MA20', 'MA50', 'RSI']].corr(), annot=True, cmap='coolwarm')
plt.title("AMAZON Feature Correlation Heatmap")
plt.show()
```

Output
![Heatmap](https://github.com/SammieBarasa77/stock_analysis/blob/main/assets/images/heatmap.png)

### Decomposition of the Time Series

**Purpose of Decomposition**

*Breaks Down Complexity: Time series decomposition separates the data into three primary components: trend, seasonality, and residual (or noise). This makes it easier to analyze and interpret patterns in the data.*

*Improves Forecasting: By isolating the trend and seasonal patterns, it allows for better prediction models since each component can be modelled separately.*

```python
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import numpy as np

decomposition = seasonal_decompose(data['Close'], model='additive', period=365)  # Assuming daily data; adjust period as needed

# Plot decomposition components
plt.figure(figsize=(12, 10))
plt.subplot(411)
plt.plot(decomposition.observed, label='Observed')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(decomposition.resid, label='Residual')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
```

Output
![Decompose](https://github.com/SammieBarasa77/stock_analysis/blob/main/assets/images/decompose.png)

### Daily Returns Distribution

```python
# Histogram
data['Daily_Returns'] = data['Close'].pct_change()
plt.figure(figsize=(8, 5))
sns.histplot(data['Daily_Returns'].dropna(), bins=50, kde=True)
plt.title("Distribution of Daily Returns")
plt.xlabel("Daily Returns")
plt.show()
The ADF Statistic
```
Output
![histogram](https://github.com/SammieBarasa77/stock_analysis/blob/main/assets/images/histogram.png)

### The ADF Statistic

```python
# Function to perform ADF test
def adf_test(series):
    result = adfuller(series.dropna())
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])
    if result[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary.")

# Perform ADF test
print("ADF Test Results for Original Data:")
adf_test(data['Close'])
```

Output
![adf](https://github.com/SammieBarasa77/stock_analysis/blob/main/assets/images/adf.png)

### Transforming the Data

From the above analysis, the data is not stationary with the P-value well above the 0.05 threshold. So, first-order differencing is often sufficient, but if the series is still not stationary.
```python
data['Close_diff'] = data['Close'].diff()
```

Testing ADF statistic again
```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(data['Close_diff'].dropna())
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
```
Output
![adf2](https://github.com/SammieBarasa77/stock_analysis/blob/main/assets/images/adf2.png)

## Train - Test Split

**Purpose of Train-Test Split**

*Model Training and Evaluation*

The training set is used to train the forecasting or machine learning model, helping it learn patterns (e.g., trends, seasonality, and relationships in the stock price data).

The test set is used to evaluate the model's performance on unseen data, simulating how the model would perform in the real world.

Ensures that the evaluation is unbiased by only testing the model on data it has not seen during training.

**Why Use an 80:20 Split?**

80% Training Data: A larger training set helps the model learn more robustly by exposing it to a wider range of patterns in the data.

20% Testing Data: This size is sufficient to provide meaningful insights into the model's performance without overly reducing the training data.

```python
# Train-test split (80:20)
train_size = int(len(data) * 0.8)
train = data[:train_size]
test = data[train_size:]
print(f"Train data size: {len(train)}")
print(f"Test data size: {len(test)}")
```
Output
![split](https://github.com/SammieBarasa77/stock_analysis/blob/main/assets/images/split.png)

```python
train
```

Output
![train_dataprvw](https://github.com/SammieBarasa77/stock_analysis/blob/main/assets/images/train_data_prevw.png)

### Preparing the Training and Testing Series

This step prepares the data for time series forecasting using the ARIMA model from statsmodels and setting up metrics for performance evaluation using mean squared error (MSE) and mean absolute error (MAE).

```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Use 'Close' price for time series forecasting
train_series = train['Close']
test_series = test['Close']
```

### Plotting the differenced data

```python
data['Price_diff'] = data['Close'].diff()

print("ADF Test Results for Differenced Data:")
adf_test(data['Price_diff'])

# Plot the differenced data
plt.figure(figsize=(12, 6))
plt.plot(data['Price_diff'], label='Differenced Data')
plt.title("Differenced Data")
plt.legend()
plt.show()
```
OutPut
![differenced_data](https://github.com/SammieBarasa77/stock_analysis/blob/main/assets/images/dfferenced_data.png)

```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Fit ARIMA model (order = (p, d, q); change based on tuning)
arima_model = ARIMA(train_series, order=(2, 4, 4))  
arima_fit = arima_model.fit()

# Forecast on the test data
forecast = arima_fit.forecast(steps=len(test_series))

# Evaluate model performance
rmse = np.sqrt(mean_squared_error(test_series, forecast))
mae = mean_absolute_error(test_series, forecast)
print(f"ARIMA Performance Metrics:\n  RMSE: {rmse:.4f}\n  MAE: {mae:.4f}")
```

### Forecasting the Model
```python
# Forecast
arima_fit = arima_model.fit()
forecast = arima_fit.forecast(steps=len(test))

# Evaluate using RMSE
rmse = np.sqrt(mean_squared_error(test_series, forecast))
print(f"RMSE: {rmse}")


# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(train['Date'], train['Close'], label='Training Data', color='blue')
plt.plot(test['Date'], test['Close'], label='Actual Test Data', color='green')
plt.plot(test['Date'], forecast, label='ARIMA Forecast', color='red')
plt.title("ARIMA Forecast vs Actual Prices")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()
```
![model_forecast](https://github.com/SammieBarasa77/stock_analysis/blob/main/assets/images/model_forecast.png)

### Hyperparameter Tuning

**Importance of Hyperparameter Tuning**

*Improves Model Performance*

*Prevents Overfitting/Underfitting*

*Enhances Generalization*

*Tailors the Model to the Dataset*

```python
from itertools import product

# Define parameter grid
p = d = q = range(0, 5)
pdq = list(product(p, d, q))

# Grid search to find the best parameters
best_score, best_params = float('inf'), None
for params in pdq:
    try:
        model = ARIMA(train_series, order=params)
        model_fit = model.fit()
        mse = mean_squared_error(test_series, model_fit.forecast(steps=len(test)))
        if mse < best_score:
            best_score, best_params = mse, params
    except:
        continue

print(f"Best Parameters: {best_params} with MSE: {best_score}")
```

Ouput
![Hyper](https://github.com/SammieBarasa77/stock_analysis/blob/main/assets/images/hyper.png)

### Regression with Random Forest

*Random Forest is a robust and versatile machine learning algorithm that can be used for regression tasks. And below is how its implemetation is done;*

```python
# Regression with Random Forests 

# Add lagged features (e.g., previous day's close price)
data['Close_1'] = data['Close'].shift(1)
data['Close_2'] = data['Close'].shift(2)

# Drop NaN rows caused by shifting
data_rf = data.dropna()

# Define features (X) and target (y)
features = ['MA20', 'MA50', 'RSI', 'Volume', 'Close_1', 'Close_2']
X = data_rf[features]
y = data_rf['Close']

# Train-test split
train_size_rf = int(len(data_rf) * 0.8)
X_train, X_test = X[:train_size_rf], X[train_size_rf:]
y_train, y_test = y[:train_size_rf], y[train_size_rf:]
```

### Training, making predictions and evaluating the Random forest regressor model

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test)

# Evaluate model
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
print(f"Random Forest Performance Metrics:\n  RMSE: {rf_rmse:.4f}\n  MAE: {rf_mae:.4f}\n  R²: {rf_r2:.4f}")
```
Output
![rsme_r2](https://github.com/SammieBarasa77/stock_analysis/blob/main/assets/images/randomf_trained_rsme_r2.png)

### Plotting Actual vs Predicted Trends

```python
# Actual vs Predicted 
plt.figure(figsize=(12, 6))
plt.plot(data_rf['Date'].iloc[train_size_rf:], y_test, label='Actual Prices', color='green')
plt.plot(data_rf['Date'].iloc[train_size_rf:], rf_predictions, label='Predicted Prices (RF)', color='orange')
plt.title("Random Forest Predictions vs Actual Prices")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()
```

![Actual vs Predicted](https://github.com/SammieBarasa77/stock_analysis/blob/main/assets/images/actual_vs_pred.png)
**Evaluating the Performance of my model in terms of percentage**
```python
from sklearn.metrics import r2_score

# Calculate R² score
r2 = r2_score(y_test, rf_predictions)

# Convert to percentage
r2_percentage = r2 * 100

print(f"R² Score: {r2:.2f}")
print(f"Model explains {r2_percentage:.2f}% of the variance in the target variable.")
```
![Pred_in%](https://github.com/SammieBarasa77/stock_analysis/blob/main/assets/images/r2_percent.png)

The R² Score of 0.93 indicates that your Random Forest regression model explains 92.96% of the variance in the target variable. This is an excellent result and demonstrates that your model captures most of the patterns in the data.

**Final Verdict**
*The model's performance is excellent, with an R² score of 0.93 showing strong predictive accuracy. With minor refinements, it can serve as a reliable tool for forecasting and analyzing trends in Amazon stock prices.*

**Presenting Findings**

Streamlit is a Python library designed to create web-based apps for machine learning and data analysis quickly and easily.

This step involves Streamlit and is crucial in terms of presenting the analysis and making insights actionable.

This step creates an interactive user interface (UI) for visualizing and sharing the stock price prediction results with others.

```python
import streamlit as st

st.title("AMZN Stock Price Prediction")

st.line_chart(data[['Close_AMZN', 'MA20_AMZN', 'MA50_AMZN']])

st.write("Predicted Close Prices:", forecast)
```
