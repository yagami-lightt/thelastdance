import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load data
df = pd.read_csv('exp7ads.csv', parse_dates=['Month'], index_col='Month')

# 1. Normal time series plot
plt.plot(df['Passengers'], label='Monthly Passengers')
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.legend()
plt.show()

# 2. ACF and PACF plots
plot_acf(df['Passengers'])
plt.title('Autocorrelation (ACF)')
plt.show()

plot_pacf(df['Passengers'])
plt.title('Partial Autocorrelation (PACF)')
plt.show()

# 3. ARIMA forecasting
model = ARIMA(df['Passengers'], order=(1, 1, 1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=12)
future_dates = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')

plt.plot(df['Passengers'], label='Observed')
plt.plot(future_dates, forecast, label='Forecasted', color='red')
plt.title('ARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.legend()
plt.show()

# 4. Print forecast
print("Next 12 months forecast:")
print(forecast)
