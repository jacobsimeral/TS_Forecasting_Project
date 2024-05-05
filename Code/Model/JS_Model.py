import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')
import random
import seaborn as sns
from sktime.split import temporal_train_test_split
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.base import ForecastingHorizon
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sktime.forecasting.arima import AutoARIMA
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import matplotlib.cm as cm
import itertools
from prophet import Prophet

def calculate_error_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
    mase = np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(np.diff(y_true)))
    return mae, mse, mape, smape, mase


def fit_prophet_and_evaluate(df_train_F, df_test_F, external_regressor=None, seasonality_params={},
                             changepoint_prior_scale=None, SPLIT_DATE=None):
    model = Prophet(
        yearly_seasonality=seasonality_params.get('yearly_seasonality', True),
        weekly_seasonality=seasonality_params.get('weekly_seasonality', True),
        daily_seasonality=seasonality_params.get('daily_seasonality', False),
        seasonality_mode=seasonality_params.get('seasonality_mode', 'additive'),
        changepoint_prior_scale=changepoint_prior_scale or 0.05
    )

    split_point = SPLIT_DATE
    df_train_F, df_test_F = df_train_F[:split_point], df_test_F[split_point:]
    if external_regressor:
        model.add_regressor(external_regressor)
    model.fit(df_train_F)

    future = model.make_future_dataframe(periods=len(df_test_F))
    future = future.tail(len(df_test_F))
    if external_regressor is not None:
        future[external_regressor] = df_test_F[external_regressor].values
    forecast = model.predict(future)
    y_test_pred = forecast['yhat'][-len(df_test_F):]
    y_true = df_test_F['y']
    y_pred = y_test_pred
    y_true, y_pred = np.array(y_true, dtype=np.float64), np.array(y_pred, dtype=np.float64)
    test_metrics = calculate_error_metrics(y_true, y_pred)
    print(
        f"Test Error Metrics: MAE: {test_metrics[0]}, MSE: {test_metrics[1]}, MAPE: {test_metrics[2]}, sMAPE: {test_metrics[3]}, MASE: {test_metrics[4]}")
    return model, forecast


def optimize_prophet(df_train, df_test, exog=None, changepoint_prior_scale_range=[0.01, 0.5, 0.1], SPLIT_DATE=None):
    best_model = None
    best_forecast = None
    best_mae = float('inf')
    best_params = {}

    param_grid = list(product(changepoint_prior_scale_range, [True, False], [True, False], [True, False],
                              ['additive', 'multiplicative']))

    for params in param_grid:
        changepoint_prior_scale = params[0]
        yearly_seasonality = params[1]
        weekly_seasonality = params[2]
        daily_seasonality = params[3]
        seasonality_mode = params[4]

        seasonality_params = {
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'daily_seasonality': daily_seasonality,
            'seasonality_mode': seasonality_mode
        }

        model, forecast = fit_prophet_and_evaluate(df_train, df_test, exog, seasonality_params, changepoint_prior_scale,
                                                   SPLIT_DATE)
        mae, _, mape, _, _ = calculate_error_metrics(df_test['y'], forecast['yhat'][-len(df_test):])

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_forecast = forecast
            best_params = {
                'changepoint_prior_scale': changepoint_prior_scale,
                'yearly_seasonality': yearly_seasonality,
                'weekly_seasonality': weekly_seasonality,
                'daily_seasonality': daily_seasonality,
                'seasonality_mode': seasonality_mode
            }
    print("Best Parameters:")
    print(best_params)
    print("Best MAE:")
    print(best_mae)
    return best_model, best_forecast


def run_stationarity_tests(dataframe, KPSS_TREND=True):
    results = []
    if KPSS_TREND:
        kp_r = 'ct'
    else:
        kp_r = 'c'
    for col in dataframe.columns:
        ts = dataframe[col]
        adf_test = adfuller(ts, autolag='AIC')
        adf_output = {
            'Test': 'ADF', 'Variable': col,
            'Test Statistic': adf_test[0],
            'p-value': adf_test[1],
            'Used Lag': adf_test[2],
            'Number of Observations': adf_test[3],
        }
        adf_output.update({f'Critical Value ({key})': value for key, value in adf_test[4].items()})
        results.append(adf_output)
        kpss_test = kpss(ts, regression=kp_r, nlags='auto')
        kpss_output = {
            'Test': 'KPSS', 'Variable': col,
            'Test Statistic': kpss_test[0],
            'p-value': kpss_test[1],
            'Used Lag': kpss_test[2],
        }
        kpss_output.update({f'Critical Value ({key})': value for key, value in kpss_test[3].items()})
        results.append(kpss_output)

    results_df = pd.DataFrame(results)

    print(
        "ADF Test Assumptions: The data has an autoregressive structure and the null hypothesis is that the time series is non-stationary or has a unit root.")
    print(
        "KPSS Test Assumptions: The null hypothesis is that the time series is stationary around a trend or constant (depending on the specified regression).")

    return results_df

def run_granger_test(data):
    granger_results = grangercausalitytests(data[['PCE', 'AHE']], maxlag=6, verbose=True)
    return granger_results


def display_granger_results(g_results):
    """
    # The null hypothesis of the Granger causality test is that past values of one time series do not provide significant information for forecasting the other time series.
    :param g_results:
    :return:graph
    """
    res_list = []
    for lag, v in g_results.items():
        res = {}
        res['Lag'] = lag
        for test, stats in v[0].items():
            res[test] = stats[1]

        res_list.append(res)

    pvals = pd.DataFrame(res_list)
    pvals.set_index('Lag', inplace=True)
    pvals_graph = pvals['params_ftest']

    pvals_graph.plot(title='Granger Causality Test')
    alpha_ser = pd.Series([0.05] * len(pvals_graph), index=pvals_graph.index)
    alpha_ser.plot(color='red')
    plt.ylabel('p-value')
    plt.show()


def plot_acf_pacf_for_dataframe(dataframe, lags=40):
    """
    Plots ACF and PACF for each column in the dataframe.

    Parameters:
    - dataframe: The dataframe with time series data in each column.
    - lags: Number of lags to include in the plots.
    """

    acf_pacf_data = {}
    print(dataframe.columns)
    for column in dataframe:
        time_series = dataframe[column]

        acf_data = acf(time_series, nlags=lags)
        pacf_data = pacf(time_series, nlags=lags, method='ywm')
        acf_pacf_data[column] = (acf_data, pacf_data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        plot_acf(time_series, lags=lags, ax=ax1)
        ax1.set_title(f'{column} - Autocorrelation Function')

        plot_pacf(time_series, lags=lags, method='ywm', ax=ax2)
        ax2.set_title(f'{column} - Partial Autocorrelation Function')

        plt.show()

    return acf_pacf_data


def recursive_forecast(time_series, original_series, trend, lambd=None, boxcox_add=0,
                       train_ratio=0.8, transformation_steps=[], seasonal_periods=12,
                       steps_ahead=12, SPLIT_DATE=None):
    """
    model_rec, preds_rec, test_rec = recursive_forecast(dataS02, data_log, trendS01,
                                                    transformation_steps=['log', 'seasonal_diff', 'subtract_trend'],
                                                    seasonal_periods=12, SPLIT_DATE='2007-01-01')
    :param time_series:
    :param original_series:
    :param trend:
    :param lambd:
    :param boxcox_add:
    :param train_ratio:
    :param transformation_steps:
    :param seasonal_periods:
    :param steps_ahead:
    :param SPLIT_DATE:
    :return:
    """
    if SPLIT_DATE is None:
        split_point = int(len(time_series) * train_ratio)
    else:
        split_point = SPLIT_DATE
    train, test = time_series[:split_point], time_series[split_point:]
    train, test = train.dropna(), test.dropna()
    original_train, original_test = original_series[:split_point], original_series[split_point:]
    trend_train, trend_test = trend[:split_point], trend[split_point:]
    values = train['co2'].values  # Extracting the values as a numpy array

    X_train = np.array([values[i:len(values) - steps_ahead + i] for i in range(steps_ahead)]).T
    y_train = values[steps_ahead:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = []
    last_values = train['co2'].values[-steps_ahead:]  # This assumes 'co2' is your target variable
    last_window = np.array(last_values).reshape(1, -steps_ahead)

    for _ in range(len(test)):
        next_step = model.predict(last_window)
        predictions.append(next_step.item())

        last_window = np.roll(last_window, -1)
        last_window[0, -1] = next_step

    predicted_mean = pd.Series(predictions, index=test.index[:len(predictions)])
    for step in reversed(transformation_steps):
        if step == 'subtract_trend':
            trend_to_add = trend_test.reindex(predicted_mean.index, method='ffill')
            predicted_mean += trend_to_add
        elif step == 'mult_100':
            predicted_mean *= 100
        elif step == 'percent_diff':
            last_original_value = original_train.iloc[-1]
            predicted_cumsum = (predicted_mean / 100 + 1).cumprod()
            predicted_mean = last_original_value * predicted_cumsum
        elif step == 'boxcox':
            predicted_mean = inv_boxcox(predicted_mean, lambd) + boxcox_add
        elif step == 'diff':
            predicted_mean = predicted_mean.cumsum() + original_train.iloc[-1]
        elif step == 'log':
            predicted_mean = np.exp(predicted_mean)
            original_train = np.exp(original_train)
            original_test = np.exp(original_test)
        elif step == 'seasonal_diff':

            for i in range(len(predicted_mean)):
                if i < seasonal_periods:
                    predicted_mean.iloc[i] += original_train.iloc[-seasonal_periods + i]
                else:
                    predicted_mean.iloc[i] += predicted_mean.iloc[i - seasonal_periods]
    plt.plot(original_train.index, original_train, color='blue', label='Train Data')
    plt.plot(original_test.index, original_test, color='orange', label='Test Data')
    plt.plot(predicted_mean.index, predicted_mean, color='green', linestyle='--', label='Predictions')
    plt.legend()
    plt.title('Actual vs Predicted')
    plt.show()
    return model, predicted_mean, original_test

