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


df = pd.read_csv("SUPERLAST.csv")

df = df[["date", "Adj Close",'gtrends','Comments_int','tweet_num','meme_Twitter','meme_Reddit']]
df = df.rename({'Adj Close': 'Adj_Close'}, axis=1)

#Step 2



for i in df.columns[1:]:
    x = df[str(i)]
    density = kde.gaussian_kde(x)
    xgrid = np.linspace(x.min(), x.max(), 100)
    plt.hist(x, bins=8,density=True, stacked=True)
    plt.plot(xgrid, density(xgrid), 'r-')
    plt.title(str(i)+ " " + "histogram" )
    plt.savefig('Plot_2/' + str(i)+"_histogram" + '.png')
    print('Plot_2/' + str(i)+"_histogram" + '.png')
    plt.show()
    



#Analog
#sns.distplot(df.gtrends, kde=True, norm_hist=True)


#Step 3

plt.boxplot(df.gtrends)
for i in df.columns[1:12]:
    plt.boxplot(df[str(i)])
    plt.title(str(i)+ " " + "boxplot" )
    plt.savefig('Plot_3/' + str(i)+"_boxplot" + '.png')
    print('Plot_3/' + str(i)+"_boxplot" + '.png')
    plt.show()  

# Step 4
data = df.gtrends

def get_best_distribution(data):
    #dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_names = ["norm", "expon", "chi2", "exponnorm", "lognorm", "gamma"]
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


for i in df.columns[1:12]:
    print(str(i) + "-" + get_best_distribution(df[str(i)])[0])
    #print(get_best_distribution(df[str(i)])[0])





# Estimating wil MLE

#Lognormal
def getLL_lognormal(params, data):
    mu,sigma,m = params
    #neg_log_lik = -np.sum(np.log(1/(data*sigma*np.sqrt(2*np.pi))) - (((np.log(data)-mu)/m)**2)/(2*sigma**2))
    neg_log_lik = -np.sum(-((np.log((data-mu)/m))**2/(2*sigma**2)) - (data - mu)*sigma*np.sqrt(2*np.pi))

    return neg_log_lik


results_ML = minimize(getLL_lognormal, [1,2,6],bounds = ((None, None), (0, None),(0, None)) , args = (df.gtrends))
mu_est_ML, sigma_est_ML, scale = results_ML.x
#by ols
get_best_distribution(df.gtrends)[2]
scipy.stats.lognorm.fit(df.gtrends)





#Gaussian
def getLL_normal(params, data):
    mu,sigma = params
    neg_log_lik = -np.sum(np.log(1/(sigma*np.sqrt(2*np.pi))) - 1/2 * ((data - mu)/sigma)**2)
    return neg_log_lik


results_ML = minimize(getLL_normal, [1,3], args = (df.meme_Twitter))
mu_est_ML, sigma_est_ML = results_ML.x
#by ols
get_best_distribution(df.meme_Twitter)[2]



#Chi-Squared

def getLL_chi2(params, data):
    df,loc,scale = params
    #neg_log_lik = -np.sum(np.log(1/(data*sigma*np.sqrt(2*np.pi))) - (((np.log(data)-mu)/m)**2)/(2*sigma**2))
    neg_log_lik = -np.sum(np.log(1/(2**(df/2)*scipy.special.gamma(df/2))) + np.log((data - loc)/scale)*(df/2-1) -((data - loc)/scale)/2  )

    return neg_log_lik

results_ML = minimize(getLL_lognormal, [1,0,1000],bounds = ((None, None), (None, None),(None, None)) , args = (df.gtrends))
mu_est_ML, sigma_est_ML, scale = results_ML.x


get_best_distribution(df.meme_Reddit)[2]


#estimating with OLS

def OLS(params, data):
    mu,sigma = params
    quantiles = [0.15,0.25,0.35,0.45,0.5,0.65,0.75,0.85,0.95]
    #s = np.random.normal(mu, sigma, 1000)
    err = 0
    for i in quantiles:
        err += (np.quantile(data, i) - norm.ppf(i, loc=mu, scale=sigma))**2
    return err




results_OLS = minimize(OLS, [12,13], args = (df.gtrends),method = 'Powell')
mu_est_OLS, sigma_est_OLS = results_OLS.x

#with standart func
norm.fit(df.gtrends)


x = np.arange(min(df.gtrends), max(df.gtrends), 1)
plt.plot(x, norm.pdf(x, mu_est_OLS, sigma_est_OLS),label="OLS")
plt.plot(x, norm.pdf(x, mu_est_ML, sigma_est_ML),label = "ML")
plt.hist(df.gtrends, density=True)
plt.legend()





# Step 5

stats.probplot(df.gtrends, dist="norm", plot=pylab)
pylab.show()


# Step 6

#kolmogorov test
stats.kstest(df.gtrends, np.random.normal(mu_est, sigma_est, 1000)  ,alternative='two-sided', mode='auto')

#Chi-Squared

chi2 = scipy.stats.chisquare(df.gtrends)

#Wilcoxon rank-sum
scipy.stats.ranksums(df.gtrends, np.random.normal(13, 12, 1000))
