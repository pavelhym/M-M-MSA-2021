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



df = pd.read_csv("SUPERLAST.csv")

df = df[["date", "Adj Close",'gtrends','Comments_int','tweet_num','meme_Twitter','meme_Reddit']]
df = df.rename({'Adj Close': 'Adj_Close'}, axis=1)

com_growth = [0]
for i in range(1,len(df)):
    if df.Comments_int.tolist()[i-1] <= df.Comments_int.tolist()[i]:
        com_growth.append(1)
    else:
        com_growth.append(0)


df["com_growth"] = com_growth
    


#Step 1
#MRV KDE

kde = stats.gaussian_kde(df.iloc[:,2:5])
density = kde(df.iloc[:,2:5])

statsmodels.nonparametric.kernel_density.KDEMultivariate(df.iloc[:,2:5],var_type = "ccc")



#Step 2
#estimation of multivariate mathematical expectation and variance.

#E
df.iloc[:,2:].apply(np.mean)

#D
df.iloc[:,2:].apply(np.std)


#Step 3
#non-parametric estimation of conditional distributions, mathematical expectations and variances.


#cond_mean

df[df['com_growth']==1].iloc[:,2:].apply(np.mean)
df[df['com_growth']==0].iloc[:,2:].apply(np.mean)


df[df['com_growth']==1].iloc[:,2:].apply(np.std)
df[df['com_growth']==0].iloc[:,2:].apply(np.std)


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



corrMatrix = df.iloc[:,2:].corr()
print(corrMatrix)




for i in df.columns[2:]:
    x = df[str(i)]
    for j in  df.columns[2:]:
        if i !=j:
            y = df[str(j)]
            corr = pearsonr_ci(y, x, alpha = 0.1)
            print('Corr of ' +  str(j) + " and " + str(i)+" is "+ str(round(corr[0],4)) + " lb-ub " + str(round(corr[2],4)) +"-"+ str(round(corr[3],4)) + " p-value= " + str(round(corr[1],4)))
     




#Step 5
#multivar corr?!?!?
for i in df.columns[2:]:
    x = df[str(i)]
    
    corr = pearsonr_ci(df['Adj_Close'], x, alpha = 0.1)
    print('Corr of  Adj_Close and ' + str(i)+" is "+ str(round(corr[0],4)) + " lb-ub " + str(round(corr[2],4)) +"-"+ str(round(corr[3],4)) + " p-value= " + str(round(corr[1],4)))
     




#Step 6
#Regression

from sklearn.model_selection import train_test_split


# Highlight predictors
X = df.iloc[:,2:7]
# Allocate the target variable
y = df[['date','Adj_Close']]
# Division into training and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

date_train = y_train['date']
date_test = y_test['date']

y_train =  y_train['Adj_Close']
y_test =  y_test['Adj_Close']


# Create a linear regression model
reg = LinearRegression(normalize=True)
# Train a linear regression model
reg.fit(X_train, y_train)
# Forecast on a test sample
y_pred = reg.predict(X_test)
params = np.append(reg.intercept_,reg.coef_)



# Calculate regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print('Mean absolute error = ', mae)
print('Mean squared error = ', mse)



def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)
print('Mean absolute percentage error = ', mape)


clf = linear_model.Lasso(alpha=0.1)
clf.fit(X_train, y_train)
print(clf.coef_)


model_aic = LassoLarsIC(criterion='aic')
model_aic.fit(X_train, y_train)
alpha_aic_ = model_aic.coef_
alpha_aic_

y_pred_lasso = clf.predict(X_test)
y_pred_lasso_aic = model_aic.predict(X_test)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mae_lasso_aic = mean_absolute_error(y_test, y_pred_lasso_aic)
print('Mean absolute error with lasso = ', mae_lasso)
print('Mean squared error with lasso = ', mse_lasso)
print('Mean absolute error with aic lasso = ', mae_lasso_aic)



#graph of real and predicted values

plt.scatter(date_test, y_test, label = u'The real meaning of Price')
plt.scatter(date_test, y_pred, label = u'Predicted by the linear model')
plt.title(u'Real values of Price')
plt.legend(loc="center right",borderaxespad=0.1, bbox_to_anchor=(1.7, 0.5))
plt.xlabel(u'date')
plt.ylabel(u'Price')


#multicoll analysis

# Building a correlation matrix of features
df.iloc[:,2:7]
corr = df.iloc[:,1:7].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corr, mask = mask, annot=True, fmt= '.1f', ax = ax, cmap = 'Blues')

#values are too correlated




#distribution of errors


res_test = y_pred -y_test


density = kde.gaussian_kde(res_test)
xgrid = np.linspace(res_test.min(), res_test.max(), 100)
plt.hist(res_test, bins=8,density=True, stacked=True)
plt.plot(xgrid, density(xgrid), 'r-')
plt.title(str(i)+ " " + "histogram" )
plt.show()


def get_best_distribution(data):
    #dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_names = ["norm", "expon",'chi2' ,"exponnorm", "lognorm", "gamma"]
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


#They are norm!
get_best_distribution(res_test)



#R**2
sklearn.metrics.r2_score(y_test, y_pred)
reg.score(X_train,y_train)
