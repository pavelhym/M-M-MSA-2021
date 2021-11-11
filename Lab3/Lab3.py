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


#Step 1

target = df[[ "Adj_Close", 'Comments_int','tweet_num']]

predictors = df.drop(columns = [ "date","Adj_Close", 'Comments_int','tweet_num'])