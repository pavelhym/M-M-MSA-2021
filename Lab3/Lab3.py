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

df = df[["date", "Adj Close",'gtrends',"Score_int",'Comments_int','tweet_num','meme_Twitter','meme_Reddit','Angry_Reddit',"Sad_Reddit"]]
df = df.rename({'Adj Close': 'Adj_Close'}, axis=1)


Q1 = df.quantile(0.2)
Q3 = df.quantile(0.8)
IQR = Q3 - Q1
df = df[~((df > (Q3 + 0.4 * IQR))).any(axis=1)]




#Step 1



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



target = df[[ "Adj_Close", 'Comments_int','tweet_num']]

predictors = df.drop(columns = [ "date","Adj_Close", 'Comments_int','tweet_num'])




get_best_distribution(df['Comments_int'])

