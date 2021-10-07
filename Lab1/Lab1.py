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


df = pd.read_csv("SUPERLAST.csv")


#Step 2

#for i in df.columns[2:12]:
#    sns.kdeplot(df[str(i)])
#    plt.hist(df[str(i)], density=True)
#    plt.title(str(i))
#    plt.show()  

for i in df.columns[2:12]:
    fig, ax = plt.subplots()
    sns.distplot(df[str(i)], bins=25, color="g", ax=ax)
    plt.title(str(i))
    plt.show()


#Step 3

plt.boxplot(df.gtrends)
for i in df.columns[2:12]:
    plt.boxplot(df[str(i)])
    plt.title(str(i)+ " " + "boxplot" )
    plt.show()  

# Step 4

dist_name = 'norm'
data = df.gtrends
def get_best_distribution(data):
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))
    print(params)
    return best_dist, best_p, params[best_dist]




# Estimating wil MLE

def getLL_normal(params, data):
    mu,sigma = params
    neg_log_lik = -np.sum(np.log(1/(sigma*np.sqrt(2*np.pi))) - 1/2 * ((data - mu)/sigma)**2)
    return neg_log_lik


guess = np.array([1,3])
results = minimize(getLL_normal, [1,3], args = (df.gtrends))

mu_est, sigma_est = results.x
#estimating with OLS



np.quantile(df['Adj Close'], 1)

# OLS

def OLS(params, data):
    mu,sigma = params
    quantiles = [0.15,0.25,0.35,0.45,0.5,0.65,0.75,0.85,0.95]
    s = np.random.normal(mu, sigma, 1000)
    err = 0
    for i in quantiles:
        err += (np.quantile(data, i) - np.quantile(s, i))**2

    return err

OLS([12,13],df.gtrends)

guess = np.array([1,3])
results = minimize(OLS, [12,13], args = (df.gtrends),method = 'Powell')



# Step 5

stats.probplot(df.gtrends, dist="norm", plot=pylab)
pylab.show()


# Step 6

#kolmogorov test
stats.kstest(df.gtrends, np.random.normal(mu_est, sigma_est, 1000)  ,alternative='two-sided', mode='auto')

#Omega squared test

