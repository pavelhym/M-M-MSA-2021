import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.formula.api import ols
import sklearn
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.optimize import minimize
import scipy.stats as stats
import pylab 
from scipy.stats import norm
from scipy.stats import kde
from scipy.stats import gamma
import scipy
from scipy import stats
import statsmodels.nonparametric.kernel_density
import copy

df = pd.read_excel("AirQualityUCI.xlsx")

for i in df.columns[2:]:
    df[str(i)].replace({-200 : None}, inplace=True)
    mean = np.mean(df[str(i)])
    df[str(i)].replace({None : mean}, inplace=True)
    df[str(i)] = pd.to_numeric(df[str(i)])




df = df.groupby(['Date']).agg({"CO(GT)": "mean",
        "PT08.S1(CO)" : "mean",
        "C6H6(GT)" : "mean",
        'PT08.S2(NMHC)' : 'mean',
        'NOx(GT)' : 'mean',
        'PT08.S3(NOx)' : 'mean',
        'NO2(GT)' : 'mean',
        'PT08.S4(NO2)' : 'mean',
        'PT08.S5(O3)' : 'mean',
        'T' : "mean",
        "RH" : 'mean',
        "AH" : "mean"
        }).reset_index()


date = df['Date']

df = df[["T", "C6H6(GT)", "AH", "PT08.S5(O3)","CO(GT)", "NO2(GT)" ]]

for i in df.columns:
    plt.plot(date, df[str(i)])
    plt.title(str(i))
    plt.savefig('Plot_1/'+ str(i) +'.png')
    plt.show()



#1
target = df[[ "T", "CO(GT)"]]

predictors = df.drop(columns = [ "T", "CO(GT)"])





# Step 1 - check stationary

import warnings
import statsmodels.api as sm
import statsmodels.tsa.api as smt
 
def tsplot(y, lags=None, figsize=(12, 7)):
 
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
 
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
 
    y.plot(ax=ts_ax)
    p_value_adf = sm.tsa.stattools.adfuller(y)[1]
    p_value_kpss = statsmodels.tsa.stattools.kpss(y,nlags="legacy")[1]
    ts_ax.set_title(str(i) +'    Dickey-Fuller: p={0:.5f}'.format(p_value_adf) + "     " +  ' KPSS: p={0:.5f}'.format(p_value_kpss))
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    plt.tight_layout()
    plt.show()
    


for i in target.columns:
    print(str(i))
    tsplot(target[str(i)])


for i in predictors.columns:
    print(str(i))
    tsplot(predictors[str(i)])


for i in df.columns:
    y = df[str(i)]
    p_value_adf = sm.tsa.stattools.adfuller(y)[1]
    p_value_kpss = statsmodels.tsa.stattools.kpss(y,nlags="legacy")[1]
    print(str(i) +  "    ADF = ",p_value_adf,"KPSS = " ,p_value_kpss  )



df_diff_temp = copy.deepcopy(df)

df_diff = pd.DataFrame()
for i in df_diff_temp.columns:
    df_diff[str(i)] = np.diff(df_diff_temp[str(i)])


for i in df_diff.columns:
    print(str(i))
    tsplot(df_diff[str(i)])
    
target = df_diff[[ "T", "CO(GT)"]]

predictors = df_diff.drop(columns = [ "T", "CO(GT)"])





from statsmodels.tsa.seasonal import seasonal_decompose

#plot for trend 

def trendline(data, order=9):
    #polynomial trend
    trend = np.polyfit(data.index.values, list(data), order)
    return np.poly1d(trend)(data.index.values)

trend = trendline(df_diff["T"], 1)
plt.plot(date[1:],df_diff["T"])
plt.plot(date[1:],trend, linewidth=4)
plt.show()

date_diff = date[1:]

#plot for seasonality

df_diff.index=pd.date_range(freq="d",start=date_diff.tolist()[0],periods=len(date_diff))
result = seasonal_decompose((df_diff["T"]), model='additive')

fig=result.plot()
fig.set_figheight(6)
fig.set_figwidth(14)

plt.show()




#Step 3 Analyze covariance or correlation function for chosen target variables and mutual correlation functions among predictors and targets.
from statsmodels.tsa import stattools







#ACF plot
statsmodels.graphics.tsaplots.plot_acf(df_diff["C6H6(GT)"])
plt.title("C6H6(GT) Autocorrelation")
plt.savefig('Plot_3/autocor_C6H6.png')
plt.show()

statsmodels.graphics.tsaplots.plot_acf(df_diff["PT08.S5(O3)"])
plt.title("PT08.S5(O3) Autocorrelation")
plt.savefig('Plot_3/autocor_PT08.png')
plt.show()

statsmodels.graphics.tsaplots.plot_acf(df_diff["CO(GT)"])
plt.title("CO(GT) Autocorrelation")
plt.savefig('Plot_3/autocor_CO.png')
plt.show()
#PACF plot
statsmodels.graphics.tsaplots.plot_pacf(df_diff["T"])

#autocov plot
plt.plot(stattools.acovf(df_diff["T"],fft=False))

#mutual correlation
import statsmodels.api as sm

#for T
alpha = 1
for i in predictors.columns:
    
    cross_corr = sm.tsa.stattools.ccf(df_diff["T"], predictors[str(i)], adjusted=False)
    plt.plot(range(0,len(cross_corr)),cross_corr, label = str(i), alpha = alpha)
    alpha -= 0.05
plt.title("Mutual corr of T")
plt.legend()
plt.savefig('Plot_3/T_mut.png')
plt.show()

#For 

alpha = 1
for i in predictors.columns:
    
    cross_corr = sm.tsa.stattools.ccf(df_diff["CO(GT)"], predictors[str(i)], adjusted=False)
    plt.plot(range(0,len(cross_corr)),cross_corr, label = str(i), alpha = alpha)
    alpha -= 0.05
plt.title("Mutual corr of CO(GT)")
plt.legend()
plt.savefig('Plot_3/CO(GT)_mut.png')
plt.show()





#Step 4 Filter high frequencies (noise) with chosen 2 filters for target variables.
from scipy import signal

window_size = 172
blackman = signal.blackman(M=window_size)
bartlett = signal.hanning(M=window_size)


f, Pxx_den = signal.welch(target["T"], fs=1, scaling='spectrum', nfft = 1000, nperseg=100)
f_window, Pxx_den_window = signal.welch(target["T"], fs=1, window = blackman, nfft = 1000, scaling='spectrum')
f_window_bart, Pxx_den_window_bart = signal.welch(target["T"], fs=1, window = bartlett, nfft = 1000, scaling='spectrum')

plt.plot(f, Pxx_den, linewidth=4)
plt.plot(f_window, Pxx_den_window)
plt.plot(f_window_bart, Pxx_den_window_bart)

plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.xlim((0,0.16))
plt.show()


f, p = signal.periodogram(x=target["T"],fs=1,window=None)
plt.plot(f,p)


#FILTERING


#!pip install fedot==0.4.1

from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.data.data_split import train_test_data_setup
from sklearn.metrics import mean_absolute_error



# Convert into numpy array first
T_time_series = np.array(target["T"])
CO_time_series = np.array(target["CO(GT)"])

# Define task - time series forecasting
# and forecast horizon 
task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=25))

T_input_data = InputData(idx=np.arange(0, len(T_time_series)),
                       features=T_time_series, target=T_time_series,
                       task=task, data_type=DataTypesEnum.ts)


CO_input_data = InputData(idx=np.arange(0, len(CO_time_series)),
                       features=CO_time_series, target=CO_time_series,
                       task=task, data_type=DataTypesEnum.ts)


#moving window

smoothing_node = PrimaryNode('smoothing') 
smoothing_node.custom_params = {'window_size': 3}

def node_fit_predict(node, input_data):
    """ Fit node and make prediction """
    node.fit(input_data)
    smoothed_output = node.predict(input_data)
    return smoothed_output

T_smoothed_mw = node_fit_predict(smoothing_node, T_input_data)
CO_smoothed_mw = node_fit_predict(smoothing_node, CO_input_data)

plt.plot(T_input_data.idx, T_input_data.target, label='Source time series')
plt.plot(T_smoothed_mw.idx, T_smoothed_mw.predict, label='Smoothed by 3 elements', linewidth=3)
plt.legend()
plt.show()

plt.plot(CO_input_data.idx, CO_input_data.target, label='Source time series')
plt.plot(CO_smoothed_mw.idx, CO_smoothed_mw.predict, label='Smoothed by 3 elements', linewidth=3)
plt.legend()
plt.show()

#Gaussian filter



gaussian_node = PrimaryNode('gaussian_filter') 
gaussian_node.custom_params = {'sigma': 2}

T_smoothed_gaussian = node_fit_predict(gaussian_node, T_input_data)
CO_smoothed_gaussian = node_fit_predict(gaussian_node, CO_input_data)

plt.plot(T_input_data.idx, T_input_data.target, label='Source time series')
plt.plot(T_smoothed_gaussian.idx, T_smoothed_gaussian.predict, label='Smoothed gaussian filter', linewidth=2, c='red')
plt.legend()
plt.show()


#Graph of two filters

plt.plot(T_input_data.idx, T_input_data.target, label='T original')
plt.plot(T_smoothed_mw.idx, T_smoothed_mw.predict, label='Smoothed by 3 elements', linewidth=3)
plt.plot(T_smoothed_gaussian.idx, T_smoothed_gaussian.predict, label='Smoothed gaussian sigma 2', linewidth=2, c='red')
plt.legend()
plt.title("T smoothing")
plt.savefig('Plot_4/T_smooth.png')
plt.show()


plt.plot(CO_input_data.idx, CO_input_data.target, label='CO(GT) original')
plt.plot(CO_smoothed_mw.idx, CO_smoothed_mw.predict, label='Smoothed by 3 elements', linewidth=3)
plt.plot(CO_smoothed_gaussian.idx, CO_smoothed_gaussian.predict, label='Smoothed gaussian sigma 2', linewidth=2, c='red')
plt.legend()
plt.title("CO(GT) smoothing")
plt.savefig('Plot_4/CO_smooth.png')
plt.show()



#Step 5 Estimate spectral density function for with and without filtering.

#for T
f, p = signal.periodogram(x=target["T"],fs=1,window=None)
f_mw, p_mw = signal.periodogram(x=T_smoothed_mw.predict,fs=1,window=None)
f_gaus, p_gaus = signal.periodogram(x=T_smoothed_gaussian.predict,fs=1,window=None)

plt.plot(f,p, label = "No smooth")
plt.plot(f_mw, p_mw, label = "Moving window")
plt.plot(f_gaus, p_gaus, label = "Gauss smoothing",c='red')
plt.legend()
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD ')
plt.title("T spectral density")
#plt.savefig('Plot_5/T_psd.png')
plt.show()


b_f, b_Pxx_den = signal.welch(target["T"], fs=1, scaling='spectrum', nfft = 1000, nperseg=100)
f, Pxx_den = signal.welch(T_smoothed_mw.predict, fs=1, scaling='spectrum', nfft = 1000, nperseg=100)
f_gauss, Pxx_den_gauss = signal.welch(T_smoothed_gaussian.predict, fs=1, scaling='spectrum', nfft = 1000, nperseg=100)

plt.plot(1/b_f, b_Pxx_den, label = 'Raw data')
plt.plot(1/f, Pxx_den, label = 'Mooving window')
plt.plot(1/f_gauss, Pxx_den_gauss, label = 'Gaussian')
plt.legend()
plt.xlim(0,20)
plt.title("T spectral density")
plt.savefig('Plot_5/T_psd.png')
plt.show()

#for CO

f_CO, p_CO = signal.periodogram(x=target["CO(GT)"],fs=1,window=None)
f_mw_CO, p_mw_CO = signal.periodogram(x=CO_smoothed_mw.predict,fs=1,window=None)
f_gaus_CO, p_gaus_CO = signal.periodogram(x=CO_smoothed_gaussian.predict,fs=1,window=None)

plt.plot(f_CO,p_CO, label = "No smooth")
plt.plot(f_mw_CO, p_mw_CO, label = "Moving window")
plt.plot(f_gaus_CO, p_gaus_CO, label = "Gauss smoothing",c='red')
plt.legend()
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD ')
plt.title("CO(GT) spectral density")
#plt.savefig('Plot_5/CO_psd.png')
plt.show()


b_f_CO, b_Pxx_den_CO = signal.welch(target["CO(GT)"], fs=1, scaling='spectrum', nfft = 1000, nperseg=100)
f_CO, Pxx_den_CO = signal.welch(CO_smoothed_mw.predict, fs=1, scaling='spectrum', nfft = 1000, nperseg=100)
f_gauss_CO, Pxx_den_gauss_CO = signal.welch(CO_smoothed_gaussian.predict, fs=1, scaling='spectrum', nfft = 1000, nperseg=100)

plt.plot(1/b_f_CO, b_Pxx_den_CO, label = 'Raw data')
plt.plot(1/f_CO, Pxx_den_CO, label = 'Mooving window')
plt.plot(1/f_gauss_CO, Pxx_den_gauss_CO, label = 'Gaussian')
plt.legend()
plt.xlim(0,20)
plt.title("CO(GT) spectral density")
plt.savefig('Plot_5/CO_psd.png')
plt.show()








#Step 6 Built auto-regression model filtered and non-filtered data. To analyze residual error and to define appropriate order of model.
#non-smooth data

train_T, test_T = train_test_data_setup(T_input_data,split_ratio= 0.8)
train_CO, test_CO = train_test_data_setup(CO_input_data,split_ratio= 0.8)
len(train_CO.idx) 
len(test_CO.idx) 



pipeline = Pipeline(PrimaryNode('ar'))
pipeline = pipeline.fine_tune_all_nodes(loss_function=mean_absolute_error,
                                        loss_params=None, input_data=train_T,
                                        iterations=500, timeout=5,
                                        cv_folds=3, validation_blocks=2)

#parameters
pipeline.print_structure()


pipeline_CO = Pipeline(PrimaryNode('ar'))
pipeline_CO = pipeline_CO.fine_tune_all_nodes(loss_function=mean_absolute_error,
                                        loss_params=None, input_data=train_CO,
                                        iterations=500, timeout=5,
                                        cv_folds=3, validation_blocks=2)


#parameters
pipeline_CO.print_structure()



fitted_vals = pipeline.fit(train_T)
forecast1d = pipeline.predict(test_T)

fitted_vals_CO = pipeline_CO.fit(train_CO)
forecast1d_CO = pipeline_CO.predict(test_CO)


plt.plot(T_input_data.idx, T_input_data.target, label='Source time series')
plt.plot(forecast1d.idx, np.ravel(forecast1d.predict), label='AR forecast')
plt.grid()
plt.legend()
plt.xlim(250,390)
plt.title("T prediction")
plt.savefig('Plot_6/T_pred.png')
plt.show()



plt.plot(CO_input_data.idx, CO_input_data.target, label='Source time series')
plt.plot(forecast1d_CO.idx, np.ravel(forecast1d_CO.predict), label='AR forecast')
plt.grid()
plt.legend()
plt.xlim(250,390)
plt.title("CO(GT) prediction")
plt.savefig('Plot_6/CO_pred.png')
plt.show()


def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


MSE_T = mean_squared_error(np.ravel(forecast1d.predict),test_T.target)
MAPE_T = mean_absolute_percentage_error(test_T.target, np.ravel(forecast1d.predict))

MSE_CO = mean_squared_error(np.ravel(forecast1d_CO.predict),test_CO.target)
MAPE_CO = mean_absolute_percentage_error(test_CO.target, np.ravel(forecast1d_CO.predict))

print("T_Predict MSE = ", MSE_T)
print("T_Predict MAPE = ", MAPE_T)

print("CO_Predict MSE = ", MSE_CO)
print("CO_Predict MAPE = ", MAPE_CO)


#smoothed data




T_time_series_sm = np.array(T_smoothed_mw.predict)
CO_time_series_sm = np.array(CO_smoothed_mw.predict)

T_input_data_sm = InputData(idx=np.arange(0, len(T_time_series_sm)),
                       features=T_time_series_sm, target=T_time_series_sm,
                       task=task, data_type=DataTypesEnum.ts)


CO_input_data_sm = InputData(idx=np.arange(0, len(CO_time_series_sm)),
                       features=CO_time_series_sm, target=CO_time_series_sm,
                       task=task, data_type=DataTypesEnum.ts)



train_T_sm, test_T_sm = train_test_data_setup(T_input_data_sm,split_ratio= 0.8)
train_CO_sm, test_CO_sm = train_test_data_setup(CO_input_data_sm,split_ratio= 0.8)




pipeline_sm = Pipeline(PrimaryNode('ar'))
pipeline_sm = pipeline_sm.fine_tune_all_nodes(loss_function=mean_absolute_error,
                                        loss_params=None, input_data=train_T_sm,
                                        iterations=500, timeout=5,
                                        cv_folds=3, validation_blocks=2)

#parameters
pipeline_sm.print_structure()


pipeline_CO_sm = Pipeline(PrimaryNode('ar'))
pipeline_CO_sm = pipeline_CO_sm.fine_tune_all_nodes(loss_function=mean_absolute_error,
                                        loss_params=None, input_data=train_CO_sm,
                                        iterations=500, timeout=5,
                                        cv_folds=3, validation_blocks=2)


#parameters
pipeline_CO_sm.print_structure()



fitted_vals_sm = pipeline_sm.fit(train_T_sm)
forecast1d_sm = pipeline_sm.predict(test_T_sm)

fitted_vals_CO_sm = pipeline_CO_sm.fit(train_CO_sm)
forecast1d_CO_sm = pipeline_CO_sm.predict(test_CO_sm)


plt.plot(T_input_data_sm.idx, T_input_data_sm.target, label='Source time series')
plt.plot(forecast1d_sm.idx, np.ravel(forecast1d_sm.predict), label='AR forecast')
plt.grid()
plt.legend()
plt.xlim(250,390)
plt.title("T smoothed prediction")
plt.savefig('Plot_6/T_pred_sm.png')
plt.show()



plt.plot(CO_input_data_sm.idx, CO_input_data_sm.target, label='Source time series')
plt.plot(forecast1d_CO_sm.idx, np.ravel(forecast1d_CO_sm.predict), label='AR forecast')
plt.grid()
plt.legend()
plt.xlim(250,390)
plt.title("CO(GT) smoothed prediction")
plt.savefig('Plot_6/CO_pred_sm.png')
plt.show()





MSE_T_sm = mean_squared_error(np.ravel(forecast1d_sm.predict),test_T_sm.target)
MAPE_T_sm = mean_absolute_percentage_error(test_T_sm.target, np.ravel(forecast1d_sm.predict))

MSE_CO_sm = mean_squared_error(np.ravel(forecast1d_CO_sm.predict),test_CO_sm.target)
MAPE_CO_sm = mean_absolute_percentage_error(test_CO_sm.target, np.ravel(forecast1d_CO_sm.predict))

print("T_Predict smoothed MSE = ", MSE_T_sm)
print("T_Predict smoothed MAPE = ", MAPE_T_sm)

print("CO_Predict smoothed MSE = ", MSE_CO_sm)
print("CO_Predict smoothed MAPE = ", MAPE_CO_sm)

#both plots
plt.plot(T_input_data.idx, T_input_data.target, label='Source time series')
plt.plot(forecast1d.idx, np.ravel(forecast1d.predict), label='AR forecast')
plt.plot(forecast1d_sm.idx, np.ravel(forecast1d_sm.predict),c = "red", label='AR smoothed forecast')
plt.grid()
plt.legend()
plt.xlim(250,390)
plt.title("T predictions comparison")
plt.savefig('Plot_6/T_pred_s+m.png')
plt.show()


plt.plot(CO_input_data.idx, CO_input_data.target, label='Source time series')
plt.plot(forecast1d_CO.idx, np.ravel(forecast1d_CO.predict), label='AR forecast')
plt.plot(forecast1d_CO_sm.idx, np.ravel(forecast1d_CO_sm.predict),c = "red", label='AR smoothed forecast')
plt.grid()
plt.legend()
plt.xlim(250,390)
plt.title("CO(GT) predictions comparison")
plt.savefig('Plot_6/CO_pred_s+m.png')
plt.show()







#with autoarima 

#from pmdarima.arima import auto_arima
#
#
#arima_model =  auto_arima(train_T.target,start_p=0, d=0, start_q=0, 
#                          max_p=5, max_d=0, max_q=5, start_P=0, 
#                          D=1, start_Q=0, max_P=5, max_D=5,
#                          max_Q=5, m=12, seasonal=False, 
#                          error_action='warn',trace = True,
#                          supress_warnings=True,stepwise = True,
#                          random_state=20,n_fits = 50 )
#
#
#prediction = arima_model.predict(n_periods = len(test_T.idx))
#
#prediction
#
#plt.plot(T_input_data.idx, T_input_data.target, label='Source time series')
#plt.plot(forecast.idx, prediction, label='AR forecast')
#plt.grid()
#plt.legend()
#plt.show()
#
#
#
##smoothed
#arima_model_sm =  auto_arima(train_sm.target,start_p=0, d=1, start_q=0, 
#                          max_p=5, max_d=5, max_q=5, start_P=0, 
#                          D=1, start_Q=0, max_P=5, max_D=5,
#                          max_Q=5, m=12, seasonal=True, 
#                          error_action='warn',trace = True,
#                          supress_warnings=True,stepwise = True,
#                          random_state=20,n_fits = 50 )
#
#
#prediction_sm = arima_model.predict(n_periods = len(test.idx))
#
#prediction_sm
#
#plt.plot(input_data_sm.idx, input_data_sm.target, label='Source time series')
#plt.plot(forecast_sm.idx, prediction_sm, label='AR forecast')
#plt.grid()
#plt.legend()
#plt.show()
#
#


##Step 7 

from fedot.core.data.multi_modal import MultiModalData

forecast_length = 25

# Data preprocessing for FEDOT
def wrap_into_input(forecast_length, feature_time_series, target_time_series):
    """ Convert data for FEDOT framework """
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))
    
    input_data = InputData(idx=np.arange(0, len(target_time_series)),
                           features=feature_time_series, target=target_time_series,
                           task=task, data_type=DataTypesEnum.ts)
    
    return input_data


#For all columns


for col_name in df_diff.columns:
    print(col_name)     


ts_T = wrap_into_input(forecast_length=forecast_length, 
                       feature_time_series=np.array(df_diff['T']),
                       target_time_series=np.array(df_diff["T"]))

ts_T_sm = wrap_into_input(forecast_length=forecast_length, 
                       feature_time_series=np.array(T_smoothed_mw.predict),
                       target_time_series=np.array(T_smoothed_mw.predict))

ts_AH = wrap_into_input(forecast_length=forecast_length, 
                       feature_time_series=np.array(df_diff['AH']),
                       target_time_series=np.array(df_diff["T"]))


ts_C6H6 = wrap_into_input(forecast_length=forecast_length, 
                       feature_time_series=np.array(df_diff['C6H6(GT)']),
                       target_time_series=np.array(df_diff["T"]))

ts_PT08 = wrap_into_input(forecast_length=forecast_length, 
                       feature_time_series=np.array(df_diff['PT08.S5(O3)']),
                       target_time_series=np.array(df_diff["T"]))


ts_NO2 = wrap_into_input(forecast_length=forecast_length, 
                       feature_time_series=np.array(df_diff['NO2(GT)']),
                       target_time_series=np.array(df_diff["T"]))


ts_CO = wrap_into_input(forecast_length=forecast_length, 
                       feature_time_series=np.array(df_diff['CO(GT)']),
                       target_time_series=np.array(df_diff["CO(GT)"]))

ts_CO_sm = wrap_into_input(forecast_length=forecast_length, 
                       feature_time_series=np.array(CO_smoothed_mw.predict),
                       target_time_series=np.array(CO_smoothed_mw.predict))



dataset = MultiModalData({
    'data_source_ts/T': ts_T,
    'data_source_ts/T_sm': ts_T_sm,
    'data_source_ts/AH': ts_AH,
    'data_source_ts/C6H6' : ts_C6H6,
    'data_source_ts/PT08' : ts_PT08,
    'data_source_ts/CO(GT)' : ts_CO,
    'data_source_ts/CO(GT)_sm' : ts_CO_sm
})

train, test = train_test_data_setup(dataset)




def create_multisource_pipeline():
    """ Generate pipeline with several data sources """
    node_source_1 = PrimaryNode('data_source_ts/T')
    node_source_2 = PrimaryNode('data_source_ts/AH')
    node_source_3 = PrimaryNode('data_source_ts/C6H6')
    node_source_4 = PrimaryNode('data_source_ts/PT08')

    node_lagged_1 = SecondaryNode('lagged', nodes_from=[node_source_1])
    node_lagged_1.custom_params = {'window_size': 20}
    node_lagged_2 = SecondaryNode('lagged', nodes_from=[node_source_2])
    
    node_lagged_3 = SecondaryNode('lagged', nodes_from=[node_source_3])
    node_lagged_4 = SecondaryNode('lagged', nodes_from=[node_source_4])

    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_lasso = SecondaryNode('lasso', nodes_from=[node_lagged_2])

    node_ridge3 = SecondaryNode('ridge', nodes_from=[node_lagged_3])
    node_lasso4 = SecondaryNode('lasso', nodes_from=[node_lagged_4])  
    
    node_final = SecondaryNode('linear', nodes_from=[node_ridge, node_lasso,node_ridge3,node_lasso4])
    pipeline = Pipeline(node_final)
    return pipeline

pipeline = create_multisource_pipeline()
pipeline.show()


pipeline.fit(train)
forecast = pipeline.predict(test)

train_length = len(df_diff) - forecast_length

plt.plot(ts_T.idx,ts_T.features, label='Actual time series')
plt.plot(np.arange(train_length, train_length + forecast_length), 
         np.ravel(forecast.predict), label='Forecast')
#plt.xlim(train_length - 100, len(df_diff['Adj_Close']) + 10)
plt.legend()
plt.grid()
plt.xlim(250,390)
plt.title("T multidata predictions")
plt.savefig('Plot_7/T_mult.png')
plt.show()


#Quality



MSE_T_multi = mean_squared_error(np.ravel(forecast.predict),test_T.target)
MAPE_T_multi = mean_absolute_percentage_error(test_T.target, np.ravel(forecast.predict))


print("T_Predict multi MSE = ", MSE_T_multi)
print("T_Predict multi MAPE = ", MAPE_T_multi)



#FOR CO


def create_multisource_pipeline_CO():
    """ Generate pipeline with several data sources """
    node_source_1 = PrimaryNode('data_source_ts/CO(GT)')
    node_source_2 = PrimaryNode('data_source_ts/AH')
    node_source_3 = PrimaryNode('data_source_ts/C6H6')
    node_source_4 = PrimaryNode('data_source_ts/PT08')

    node_lagged_1 = SecondaryNode('lagged', nodes_from=[node_source_1])
    node_lagged_1.custom_params = {'window_size': 20}
    node_lagged_2 = SecondaryNode('lagged', nodes_from=[node_source_2])
    node_lagged_1.custom_params = {'window_size': 3}
    node_lagged_3 = SecondaryNode('lagged', nodes_from=[node_source_3])
    node_lagged_4 = SecondaryNode('lagged', nodes_from=[node_source_4])

    node_ridge = SecondaryNode('lasso', nodes_from=[node_lagged_1])
    node_lasso = SecondaryNode('lasso', nodes_from=[node_lagged_2])

    #node_ridge3 = SecondaryNode('lasso', nodes_from=[node_lagged_3])
    #node_lasso4 = SecondaryNode('lasso', nodes_from=[node_lagged_4])  
    
    node_final = SecondaryNode('linear', nodes_from=[node_lagged_1,node_lagged_2 ])
    pipeline = Pipeline(node_final)
    return pipeline


pipeline_CO = create_multisource_pipeline_CO()
pipeline_CO.show()


pipeline_CO.fit(train)
forecast_CO = pipeline_CO.predict(test)

train_length = len(df_diff) - forecast_length

plt.plot(ts_CO.idx,ts_CO.features, label='Actual time series')
plt.plot(np.arange(train_length, train_length + forecast_length), 
         np.ravel(forecast_CO.predict), label='Forecast')
#plt.xlim(train_length - 100, len(df_diff['Adj_Close']) + 10)
plt.legend()
plt.grid()
plt.xlim(250,390)
plt.title("CO multidata predictions")
plt.savefig('Plot_7/CO_mult.png')
plt.show()



MSE_CO_multi = mean_squared_error(np.ravel(forecast_CO.predict),test_CO.target)
MAPE_CO_multi = mean_absolute_percentage_error(test_CO.target, np.ravel(forecast_CO.predict))


print("CO_Predict multi MSE = ", MSE_CO_multi)
print("CO_Predict multi MAPE = ", MAPE_CO_multi)




#For Smoothed data 


def create_multisource_pipeline_sm():
    """ Generate pipeline with several data sources """
    node_source_1 = PrimaryNode('data_source_ts/T_sm')
    node_source_2 = PrimaryNode('data_source_ts/AH')
    node_source_3 = PrimaryNode('data_source_ts/C6H6')
    node_source_4 = PrimaryNode('data_source_ts/PT08')

    node_lagged_1 = SecondaryNode('lagged', nodes_from=[node_source_1])
    node_lagged_1.custom_params = {'window_size': 20}
    node_lagged_2 = SecondaryNode('lagged', nodes_from=[node_source_2])
    node_lagged_2.custom_params = {'window_size': 10}

    node_lagged_3 = SecondaryNode('lagged', nodes_from=[node_source_3])
    node_lagged_3.custom_params = {'window_size': 10}
    node_lagged_4 = SecondaryNode('lagged', nodes_from=[node_source_4])
    node_lagged_4.custom_params = {'window_size': 10}

    node_ridge = SecondaryNode('lasso', nodes_from=[node_lagged_1])
    node_lasso = SecondaryNode('lasso', nodes_from=[node_lagged_2])

    node_ridge3 = SecondaryNode('lasso', nodes_from=[node_lagged_3])
    node_lasso4 = SecondaryNode('lasso', nodes_from=[node_lagged_4])  
    
    node_final = SecondaryNode('linear', nodes_from=[node_ridge, node_lasso,node_ridge3,node_lasso4])
    pipeline = Pipeline(node_final)
    return pipeline

pipeline_sm = create_multisource_pipeline_sm()
pipeline_sm.show()


pipeline_sm.fit(train)
forecast_sm = pipeline_sm.predict(test)

train_length = len(df_diff) - forecast_length

plt.plot(ts_T_sm.idx,ts_T_sm.features, label='Actual time series')
plt.plot(np.arange(train_length, train_length + forecast_length), 
         np.ravel(forecast_sm.predict), label='Forecast')
#plt.xlim(train_length - 100, len(df_diff['Adj_Close']) + 10)
plt.legend()
plt.grid()
plt.xlim(250,390)
plt.title("T multidata smoothed predictions")
plt.savefig('Plot_7/T_mult_sm.png')
plt.show()


#Quality



MSE_T_multi_sm = mean_squared_error(np.ravel(forecast_sm.predict),test_T_sm.target)
MAPE_T_multi_sm = mean_absolute_percentage_error(test_T_sm.target, np.ravel(forecast_sm.predict))
print("T_Predict smoothed multi MSE = ", MSE_T_multi_sm)
print("T_Predict smoothed multi MAPE = ", MAPE_T_multi_sm)


#For CO_sm

def create_multisource_pipeline_CO_sm():
    """ Generate pipeline with several data sources """
    node_source_1 = PrimaryNode('data_source_ts/CO(GT)_sm')
    node_source_2 = PrimaryNode('data_source_ts/AH')
    node_source_3 = PrimaryNode('data_source_ts/C6H6')
    node_source_4 = PrimaryNode('data_source_ts/PT08')

    node_lagged_1 = SecondaryNode('lagged', nodes_from=[node_source_1])
    node_lagged_1.custom_params = {'window_size': 10}
    node_lagged_2 = SecondaryNode('lagged', nodes_from=[node_source_2])
    node_lagged_2.custom_params = {'window_size': 1}
    node_lagged_3 = SecondaryNode('lagged', nodes_from=[node_source_3])
    node_lagged_4 = SecondaryNode('lagged', nodes_from=[node_source_4])

    node_ridge = SecondaryNode('lasso', nodes_from=[node_lagged_1])
    node_lasso = SecondaryNode('lasso', nodes_from=[node_lagged_2])

    #node_ridge3 = SecondaryNode('lasso', nodes_from=[node_lagged_3])
    #node_lasso4 = SecondaryNode('lasso', nodes_from=[node_lagged_4])  
    
    node_final = SecondaryNode('linear', nodes_from=[node_lagged_1,node_lagged_2 ])
    pipeline = Pipeline(node_final)
    return pipeline


pipeline_CO_sm = create_multisource_pipeline_CO_sm()
pipeline_CO_sm.show()


pipeline_CO_sm.fit(train)
forecast_CO_sm = pipeline_CO_sm.predict(test)

train_length = len(df_diff) - forecast_length

plt.plot(ts_CO_sm.idx,ts_CO_sm.features, label='Actual time series')
plt.plot(np.arange(train_length, train_length + forecast_length), 
         np.ravel(forecast_CO_sm.predict), label='Forecast')
#plt.xlim(train_length - 100, len(df_diff['Adj_Close']) + 10)
plt.legend()
plt.grid()
plt.xlim(250,390)
plt.title("CO multidata smoothed predictions")
plt.savefig('Plot_7/CO_mult_sm.png')
plt.show()



MSE_CO_multi_sm = mean_squared_error(np.ravel(forecast_CO_sm.predict),test_CO_sm.target)
MAPE_CO_multi_sm = mean_absolute_percentage_error(test_CO_sm.target, np.ravel(forecast_CO_sm.predict))


print("CO_Predict smoothed multi MSE = ", MSE_CO_multi_sm)
print("CO_Predict smoothed multi MAPE = ", MAPE_CO_multi_sm)



plt.plot(T_input_data.idx, T_input_data.target, label='Source time series')
plt.plot(forecast1d.idx, np.ravel(forecast1d.predict), label='AR forecast')
plt.plot(np.arange(train_length, train_length + forecast_length), 
         np.ravel(forecast.predict), label='Forecast Multi')
plt.grid()
plt.legend()
plt.xlim(250,390)
plt.title("T predictions comparison")
plt.savefig('Plot_7/T_pred_1d+mult.png')
plt.show()


plt.plot(CO_input_data.idx, CO_input_data.target, label='Source time series')
plt.plot(forecast1d_CO.idx, np.ravel(forecast1d_CO.predict), label='AR forecast')
plt.plot(np.arange(train_length, train_length + forecast_length), 
         np.ravel(forecast_CO.predict), label='Forecast')
plt.grid()
plt.legend()
plt.xlim(250,390)
plt.title("CO(GT) predictions comparison")
plt.savefig('Plot_7/CO_pred_1d+mult.png')
plt.show()
