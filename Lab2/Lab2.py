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


import numpy as np
import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import math
import pylab 
from sklearn.linear_model import  LassoLarsIC

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

df = df[["T", "RH", "AH", "PT08.S5(O3)","CO(GT)", "NO2(GT)" ]]


#Step 1
#MRV KDE

kde = stats.gaussian_kde(df.iloc[:,2:5])
density = kde(df.iloc[:,2:5])

statsmodels.nonparametric.kernel_density.KDEMultivariate(df,var_type = "cccccc")

sns.pairplot(df, kind="kde")
plt.savefig('Plot_1/PDF.png')
plt.show()
#Step 2
#estimation of multivariate mathematical expectation and variance.

#E
df.apply(np.mean)

#stD
df.apply(np.std)


#Step 3
#non-parametric estimation of conditional distributions, mathematical expectations and variances.

#create new dummy varible 
AH_dummy = []
for i in range(len(df)):
    if df['AH'][i] > np.median(df['AH']):
        AH_dummy.append(1)
    else:
        AH_dummy.append(0)

df_dum = copy.deepcopy(df)
df_dum['AH_dummy'] = AH_dummy

from scipy.stats import kde
for i in df.columns[0:]:
    if i == "AH":
        continue
    for cond in [0,1]:

        x = df_dum[df_dum["AH_dummy"]==cond][str(i)].tolist()
        density = kde.gaussian_kde(x)
        xgrid = np.linspace(min(x), max(x), 100)
        plt.hist(x, bins=8,density=True, stacked=True)
        plt.plot(xgrid, density(xgrid), 'r-')
        plt.title(str(i)+ " " + "histogram" + " AH_dummy = " + str(cond) )
        plt.savefig('Plot_3/' + str(i)+"_histogram" + str(cond) +  '.png')
        plt.show()





round(2.45,1)
#cond_mean

round(df_dum[df_dum["AH_dummy"]==1].apply(np.mean),2)
round(df_dum[df_dum["AH_dummy"]==0].apply(np.mean),2)


#cond std
round(df_dum[df_dum["AH_dummy"]==1].apply(np.std),3)
round(df_dum[df_dum["AH_dummy"]==0].apply(np.std),3)



#Step 4
#estimation of pair correlation coefficients, confidence intervals for them and significance levels.


def pearsonr_ci(x,y,alpha=0.05):
    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

pearsonr_ci(df['Adj_Close'], df['gtrends'], alpha = 0.1)



corrMatrix = df.corr()
print(corrMatrix)




for i in df.columns:
    x = df[str(i)]
    for j in  df.columns:
        if i !=j:
            y = df[str(j)]
            corr = pearsonr_ci(y, x, alpha = 0.1)
            print('Corr of ' +  str(j) + " and " + str(i)+" is "+ str(round(corr[0],4)) + " lb-ub " + str(round(corr[2],4)) +"/"+ str(round(corr[3],4)) + " p-value= " + str(round(corr[1],4)))
     




#Step 5
#multivar corr?!?!?
for i in df.columns[2:]:
    x = df[str(i)]
    
    corr = pearsonr_ci(df['Adj_Close'], x, alpha = 0.1)
    print('Corr of  Adj_Close and ' + str(i)+" is "+ str(round(corr[0],4)) + " lb-ub " + str(round(corr[2],4)) +"-"+ str(round(corr[3],4)) + " p-value= " + str(round(corr[1],4)))
     




#Step 6
#Regression

from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
import pyperclip as pc

df.columns.tolist()
df.columns = ['T', 'RH', 'AH', 'PT08_S5', 'CO_GT', 'NO2_GT']
#Data preparation
X = df.iloc[:,1:]
# Allocate the target variable
df['date'] = date
y = df[['date','T']]
# Division into training and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
date_train = y_train['date']
date_test = y_test['date']
y_train =  y_train['T']
y_test =  y_test['T']

df_train , df_test =  train_test_split(df, test_size=0.33, random_state=42)


#OLS MODEL with nice summary
ols_ws = ols('T ~ RH + AH + PT08_S5 + CO_GT + NO2_GT  ', data=df_train).fit()

pc.copy(ols_ws.summary().as_latex())

y_pred = ols_ws.predict(df_test)
y_pred_train = ols_ws.predict(df_train)

#OLS MODEL from workshop
# Create a linear regression model
ols_reg = LinearRegression()
# Train a linear regression model
ols_reg.fit(X_train, y_train)
# Forecast on a test sample
#y_pred = ols_reg.predict(X_test)



# Calculate regression metrics
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


mae_ols = mean_absolute_error(y_test, y_pred)
mse_ols = mean_squared_error(y_test, y_pred)
mape_ols = mean_absolute_percentage_error(y_test, y_pred)
print('MAE = ', mae_ols)
print('MSE = ',  mse_ols )
print('MAPE = ', mape_ols)


mae_ols_train = mean_absolute_error(y_train, y_pred_train)
mse_ols_train = mean_squared_error(y_train, y_pred_train)
mape_ols_train = mean_absolute_percentage_error(y_train, y_pred_train)
print('MAE train = ', mae_ols_train)
print('MSE train = ',  mse_ols_train )
print('MAPE train = ', mape_ols_train)




#Lasso regularization
clf = linear_model.Lasso(alpha=0.1)
clf.fit(X_train, y_train)

y_pred_lasso = clf.predict(X_test)
y_pred_lasso_train = clf.predict(X_train)


#Lasso with tuned parameter
model_aic = LassoLarsIC(criterion='aic')
model_aic.fit(X_train, y_train)
alpha_aic_ = model_aic.coef_
alpha_aic_
y_pred_lasso_aic = model_aic.predict(X_test)
y_pred_lasso_aic_train = model_aic.predict(X_train)

#on test
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mape_lasso = mean_absolute_percentage_error(y_test, y_pred_lasso)
mae_lasso_aic = mean_absolute_error(y_test, y_pred_lasso_aic)
mse_lasso_aic = mean_squared_error(y_test, y_pred_lasso_aic)
mape_lasso_aic = mean_absolute_percentage_error(y_test, y_pred_lasso_aic)


print('MAE = ', mae_ols)
print('MSE  = ',  mse_ols )
print('MAPE = ', mape_ols)
print('MAE with lasso = ', mae_lasso)
print('MSE with lasso = ', mse_lasso)
print('MAPE with lasso = ', mape_lasso)
print('MAE with aic lasso = ', mae_lasso_aic)
print('MSE with aic lasso = ', mse_lasso_aic)
print('MAPE with aic lasso = ', mape_lasso_aic)





#on train 
mae_lasso_train = mean_absolute_error(y_train, y_pred_lasso_train)
mse_lasso_train = mean_squared_error(y_train, y_pred_lasso_train)
mape_lasso_train = mean_absolute_percentage_error(y_train, y_pred_lasso_train)
mae_lasso_aic_train = mean_absolute_error(y_train, y_pred_lasso_aic_train)
mse_lasso_aic_train = mean_squared_error(y_train, y_pred_lasso_aic_train)
mape_lasso_aic_train = mean_absolute_percentage_error(y_train, y_pred_lasso_aic_train)

print('MAE train = ', mae_ols_train)
print('MSE train = ',  mse_ols_train )
print('MAPE train = ', mape_ols_train)
print('MAE with lasso train = ', mae_lasso_train)
print('MSE with lasso train = ', mse_lasso_train)
print('MAPE with lasso train = ', mape_lasso_train)
print('MAE with aic lasso train = ', mae_lasso_aic_train)
print('MSE with aic lasso train = ', mse_lasso_aic_train)
print('MAPE with aic lasso train = ', mape_lasso_aic_train)







#graph of real and predicted values

plt.scatter(date_test, y_test, label = u'The real T')
plt.scatter(date_test, y_pred, label = u'Predicted by the linear model')
plt.title(u'Prediction')
plt.legend(loc="center right",borderaxespad=0.1, bbox_to_anchor=(1.7, 0.5))
plt.xlabel(u'date')
plt.ylabel(u'Temperature')


#multicoll analysis

# Building a correlation matrix of features
corr = df.iloc[:,1:6].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(10, 10))

map =  sns.heatmap(corr, mask = mask, annot=True, fmt= '.1f', ax = ax, cmap = 'Blues')

fig = map.get_figure()
fig.savefig("Plot_6/corr_map.png")


from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X_train.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
  
print(vif_data)



#values are too correlated

#smaller OLS
ols_ws_small = ols('T ~ RH + AH + NO2_GT  ', data=df_train).fit()

pc.copy(ols_ws_small.summary().as_latex())

y_pred_small = ols_ws_small.predict(df_test)
y_pred_train_small = ols_ws_small.predict(df_train)


mae_ols_small = mean_absolute_error(y_test, y_pred_small)
mse_ols_small = mean_squared_error(y_test, y_pred_small)
mape_ols_small = mean_absolute_percentage_error(y_test, y_pred_small)
print('MAE small = ', mae_ols_small)
print('MSE small = ',  mse_ols_small )
print('MAPE small = ', mape_ols_small)


mae_ols_train_small = mean_absolute_error(y_train, y_pred_train_small)
mse_ols_train_small = mean_squared_error(y_train, y_pred_train_small)
mape_ols_train_small = mean_absolute_percentage_error(y_train, y_pred_train_small)
print('MAE train small = ', mae_ols_train_small)
print('MSE train small = ',  mse_ols_train_small )
print('MAPE train small = ', mape_ols_train_small)


#PCA analysis
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
pca = PCA(n_components=2)

X_train_PCA = StandardScaler().fit_transform(X_train)
X_test_PCA = StandardScaler().fit_transform(X_test)




principalComponents_train = pca.fit_transform(X_train_PCA)
principalComponents_test = pca.fit_transform(X_test_PCA)


df_train_PCA = pd.DataFrame(data = principalComponents_train, columns = ['PC1', 'PC2'])
df_train_PCA["T"] = y_train.tolist()

df_test_PCA = pd.DataFrame(data = principalComponents_test, columns = ['PC1', 'PC2'])
df_test_PCA["T"] = y_test.tolist()


#OLS MODEL
ols_ws_PCA = ols('T ~ PC1 + PC2  ', data=df_train_PCA).fit()

pc.copy(ols_ws_PCA.summary().as_latex())

y_pred_PCA = ols_ws_PCA.predict(df_test_PCA)
y_pred_train_PCA = ols_ws_PCA.predict(df_train_PCA)


mae_ols_PCA = mean_absolute_error(y_test, y_pred_PCA)
mse_ols_PCA = mean_squared_error(y_test, y_pred_PCA)
mape_ols_PCA = mean_absolute_percentage_error(y_test, y_pred_PCA)
print('MAE  PCA = ', mae_ols_PCA)
print('MSE  PCA = ',  mse_ols_PCA )
print('MAPE PCA = ', mape_ols_PCA)


mae_ols_train_PCA = mean_absolute_error(y_train, y_pred_train_PCA)
mse_ols_train_PCA = mean_squared_error(y_train, y_pred_train_PCA)
mape_ols_train_PCA = mean_absolute_percentage_error(y_train, y_pred_train_PCA)
print('MAE  train PCA = ', mae_ols_train_PCA)
print('MSE  train PCA = ',  mse_ols_train_PCA )
print('MAPE train PCA = ', mape_ols_train_PCA)




#step 7

#model comp

#OLS
print("R2 OLS = ",ols_ws.rsquared)
print("AIC OLS = ", ols_ws.aic)
print('MSE train = ',  mse_ols_train )
print('MSE test  = ',  mse_ols )

#OLS short

print("R2 OLS small = ",ols_ws_small.rsquared)
print("AIC OLS small = ", ols_ws_small.aic)
print('MSE train small = ',  mse_ols_train_small )
print('MSE  = small test ',  mse_ols_small )

#Lasso
def AIC(y_true, y_pred, k):
    n = len(y_true)
    RSS = sum((y_true - y_pred)**2)
    AIC = 2*k + n*np.log(RSS)
    return AIC

AIC(y_train, y_pred_train, 5)

print("R2 OLS lasso = ", sklearn.metrics.r2_score(y_train, y_pred_lasso_train))
print("AIC lasso = ", AIC(y_train, y_pred_lasso_train, 5))
print('MSE lasso train  = ',  mse_lasso_train )
print('MSE lasso test  = small ',  mse_lasso )

#Lasso AIC
print("R2 OLS lasso AIC = ", sklearn.metrics.r2_score(y_train, y_pred_lasso_aic_train))
print("AIC lasso AIC = ", AIC(y_train, y_pred_lasso_aic_train, 5))
print('MSE with aic lasso train = ', mse_lasso_aic_train)
print('MSE with aic lasso = ', mse_lasso_aic)

#PCA

print("R2 OLS PCA = ",ols_ws_PCA.rsquared)
print("AIC OLS PCA = ", ols_ws_PCA.aic)
print('MSE train PCA = ',  mse_ols_train_PCA )
print('MSE test PCA  = ',  mse_ols_PCA )


#distribution of errors


res_test = y_pred - y_test
res_train = y_pred_train - y_train


from scipy.stats import kde
density = kde.gaussian_kde(res_train)
xgrid = np.linspace(res_train.min(), res_train.max(), 100)
plt.hist(res_train, bins=8,density=True, stacked=True)
plt.plot(xgrid, density(xgrid), 'r-')
plt.title("Train residials"+ " " + "histogram" )
plt.savefig('Plot_7/Train_hist.png')
plt.show()

from scipy.stats import kde
density = kde.gaussian_kde(res_test)
xgrid = np.linspace(res_test.min(), res_test.max(), 100)
plt.hist(res_test, bins=8,density=True, stacked=True)
plt.plot(xgrid, density(xgrid), 'r-')
plt.title("Test residials"+ " " + "histogram" )
plt.savefig('Plot_7/Test_hist.png')
plt.show()


def get_best_distribution(data):
    #dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_names = ["norm"]
    #dist_name = "lognorm"
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        #print("p value for "+dist_name+" = "+str(p))
        
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    #print("Best fitting distribution: "+str(best_dist))
    #print("Best p value: "+ str(best_p))
    #print("Parameters for the best fit: "+ str(params[best_dist]))
    #print(params)
    return best_dist, best_p, params[best_dist]


#They are almost norm!
get_best_distribution(res_train)


#graph of real and predicted values

plt.scatter(date_test, y_test, label = u'The real T')
plt.scatter(date_test, y_pred, label = u'Predicted by the linear model')
plt.title(u'Prediction by OLS')
plt.legend(loc="center right",borderaxespad=0.1, bbox_to_anchor=(1.7, 0.5))
plt.xlabel(u'date')
plt.savefig('Plot_7/pred.png')
plt.ylabel(u'Temperature')

