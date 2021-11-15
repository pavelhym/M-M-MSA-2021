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

#Step 2
#inverse sampling
#here our inverse sampling with expon func
def inv_samp_exp(loc,s):
  x = np.random.uniform()
  return -loc - s*math.log(1-x)


#rejection sampling

#make some dict with pdf
dist_name_to_func_pdf = {
    "norm" : (lambda i, j : scipy.stats.norm.pdf(i, *j)),
    "exponnorm" : (lambda i, j: scipy.stats.exponnorm.pdf(i, *j)),
    "genextreme" : (lambda i, j: scipy.stats.genextreme.pdf(i, *j)),
    "expon" : (lambda i, j: scipy.stats.expon.pdf(i, *j)),
    "chi2" : (lambda i, j: scipy.stats.chi2.pdf(i, *j)),
    "lognorm" : (lambda i, j: scipy.stats.lognorm.pdf(i, *j)),
    "gamma" : (lambda i, j: scipy.stats.gamma.pdf(i, *j)),
    "exponweib" : (lambda i, j: scipy.stats.exponweib.pdf(i, *j)),
    "weibull_max" : (lambda i, j: scipy.stats.weibull_max.pdf(i, *j)),
    "weibull_min" : (lambda i, j: scipy.stats.weibull_min.pdf(i, *j)),
    "pareto" : (lambda i, j: scipy.stats.pareto.pdf(i, *j))

}

def brute_forse(func, a, b, eps):
  bigest_so_far = a
  max_func_value = func(a)
  n = math.ceil((b -a)/eps)
  for k in range(1, n+1):
    x = a + k*(b-a)/n
    fx = func(x)
    if fx>max_func_value:
      bigest_so_far = x
      max_func_value=fx
  return max_func_value

def rejection_samp(p, q, q_sample, max_x = 800000, iterations = 10**5):
  number_of_steps = 100000
  M = brute_forse((lambda x: p(x)/q(x)), 0, max_x, max_x/number_of_steps)
  #M=5
  #collect all accepted samples here
  samples = []

  #try this many candidates
  N = iterations

  for _ in range(N):
      #sample a candidate
      candidate = q_sample()
      
      #calculate probability of accepting this candidate
      prob_accept = p(candidate) / (M*q(candidate))
      
      #accept with the calculated probability
      if np.random.random() < prob_accept:
          samples.append(candidate)
  return samples

for i in target:
    x = df[str(i)]
    density = kde.gaussian_kde(x) #p function(for rejection sampling)
    app_name, app_pvalue, app_params = get_best_distribution(df[str(i)])
    app_density = lambda y: dist_name_to_func_pdf[app_name](y, app_params) #q 
    samples = rejection_samp(density, app_density, lambda: scipy.stats.genextreme.rvs(*app_params)) #we use non-comon dist here!!!
    xgrid = np.linspace(x.min(), x.max(), 100)
    #plt.hist(x, bins=8,density=True, stacked=True)
    print(samples)
    plt.hist(samples, bins=1000,density=True, stacked=True)
    plt.plot(xgrid, density(xgrid), 'r-')
    plt.title(str(i)+ " " + "histogram" )
    plt.show()
