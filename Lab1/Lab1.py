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


for i in df.columns[1:12]:
    print(str(i) + "-" + get_best_distribution(df[str(i)])[0])
    #print(get_best_distribution(df[str(i)])[0])



df.columns

# Estimating wil MLE

#Lognormal
for names in ['Adj_Close', 'gtrends','Comments_int',"tweet_num"]:
    var = df[str(names)]
    def getLL_lognormal(params, data):
        #mu,sigma,m = params
        #neg_log_lik = -np.sum(np.log(1/(data*sigma*np.sqrt(2*np.pi))) - (((np.log(data)-mu)/m)**2)/(2*sigma**2))
        #neg_log_lik = -np.sum(-((np.log((data-mu)/m))**2/(2*sigma**2)) - (data - mu)*sigma*np.sqrt(2*np.pi))
        s,loc,scale = params
        x = (data - loc)/scale
        neg_log_lik = -np.sum(-np.log(s*x*np.sqrt(2*np.pi)) - np.log(x)**2/(2*s**2) - np.log(scale))
        return neg_log_lik 


    results_ML_lognorm = minimize(getLL_lognormal, [x+1 for x in list(scipy.stats.lognorm.fit(var))],bounds = ((None, None), (0.000000001, None),(0.000000001, None)) , args = (var))
    mu_est_ML, sigma_est_ML, scale_est_ML = results_ML_lognorm.x
    #by ML from package
    ols_params = get_best_distribution(var)[2]
    list(scipy.stats.lognorm.fit(var))
    [x+1 for x in list(scipy.stats.lognorm.fit(var))]



    #by OLS
    def OLS_lognorm(params, data):
        s,loc,scale = params
        quantiles = [0.15,0.25,0.35,0.45,0.5,0.65,0.75,0.85,0.95]
        #s = np.random.normal(mu, sigma, 1000)
        err = 0
        for int in range(0,100,5):
            i = int/100
            #err += (np.quantile(data, i) - norm.ppf(i, loc=mu, scale=sigma))**2
            err += (i - scipy.stats.lognorm.cdf(np.quantile(data, i), s, loc,scale))**2
        return err

    results_OLS_lognorm = minimize(OLS_lognorm, [x+1 for x in list(scipy.stats.lognorm.fit(var))],bounds = ((None, None), (0.000001, None),(0.000000001, None)) , args = (var))
    mu_est_OLS, sigma_est_OLS, scale_est_OLS = results_OLS_lognorm.x

    print(str(names))
    print("Results by package ML:")
    print(scipy.stats.lognorm.fit(var))
    print("Results by handmade ML:")
    print(results_ML_lognorm.x)

    print("Results by handmade OLS:")
    print(results_OLS_lognorm.x)







#Gaussian
def getLL_normal(params, data):
    mu,sigma = params
    neg_log_lik = -np.sum(np.log(1/(sigma*np.sqrt(2*np.pi))) - 1/2 * ((data - mu)/sigma)**2)
    return neg_log_lik


results_ML_gauss = minimize(getLL_normal, [1,3], args = (df.meme_Twitter))
mu_est_ML, sigma_est_ML = results_ML_gauss.x
#by ML from package
get_best_distribution(df.meme_Twitter)[2]

#by OLS

def OLS_gauss(params, data):
    mu,sigma = params
    quantiles = [0.15,0.25,0.35,0.45,0.5,0.65,0.75,0.85,0.95]
    #s = np.random.normal(mu, sigma, 1000)
    err = 0
    #for int in range(0,100,5):
    for i in quantiles:
        #i = int/100
        err += (np.quantile(data, i) - norm.ppf(i, loc=mu, scale=sigma))**2
        #err += (i - norm.cdf(np.quantile(data, i), mu, sigma))**2
    return err


results_OLS_gauss = minimize(OLS_gauss, [2,10], args = (df.meme_Twitter),bounds = ((None, None), (0.000000001, None)),method = 'Powell')
mu_est_OLS, sigma_est_OLS = results_OLS_gauss.x

print("Results by handmade ML:")
print(results_ML_gauss.x)
print("Results by package ML:")
print(get_best_distribution(df.meme_Twitter)[2])
print("Results by handmade OLS:")
print(results_OLS_gauss.x)



#Chi-Squared

meme_Reddit =  df.meme_Reddit

def getLL_chi2(params, data):
    df,loc,scale = params
    #neg_log_lik = -np.sum(np.log(1/(data*sigma*np.sqrt(2*np.pi))) - (((np.log(data)-mu)/m)**2)/(2*sigma**2))
    
    x = (data - loc)/scale
    #print(scale)
    #print(min(x))

    #neg_log_lik = -np.sum(-np.log(2**(df/2)) - np.log(scipy.special.gamma(df/2)) + (df/2-1)*np.log(x) - x/2 - np.log(scale))
    neg__lik = -np.prod(1/(2**(df/2) * scipy.special.gamma(df/2))*x**(df/2-1)*np.exp(-x/2)/scale)
    return neg__lik

results_ML_chi2 = minimize(getLL_chi2, [0,-1,1000],bounds = ((0, 1), (-1, 1),(1000, 1200)) , args = (meme_Reddit),method = 'Powell')
mu_est_ML, sigma_est_ML, scale = results_ML_chi2.x

#by ML from package
scipy.stats.chi2.fit(meme_Reddit)

#by OLS
def OLS_chi2(params, data):
    s,loc,scale = params
    quantiles = [0.15,0.25,0.35,0.45,0.5,0.65,0.75,0.85,0.95]
    #s = np.random.normal(mu, sigma, 1000)
    err = 0
    for int in range(0,100,2):
        i = int/100
        #err += (np.quantile(data, i) - norm.ppf(i, loc=mu, scale=sigma))**2
        err += (i - scipy.stats.chi2.cdf(np.quantile(data, i), s, loc,scale))**2
    return err

results_OLS_chi2 = minimize(OLS_chi2, [0,-1,1000],bounds = ((0, 1), (-1, 1),(1000, 1200)), args = (df.meme_Reddit),method = 'Powell')
s_est_OLS,loc_est_OLS,scale_est_OLS = results_OLS_chi2.x


print("Results by handmade ML:")
print(results_ML_chi2.x)
print("Results by package ML:")
print(scipy.stats.chi2.fit(meme_Reddit))
print("Results by handmade OLS:")
print(results_OLS_chi2.x)






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
stats.kstest(df.gtrends, np.random.normal(1, 1, 1000)  ,alternative='two-sided', mode='auto')
stats.kstest(df.gtrends.tolist(), np.random.normal(0.8758370472673307, 2.5450828743821496, 1000),alternative='two-sided', mode='auto')
#Chi-Squared

chi2 = scipy.stats.chisquare(df.gtrends)

#Wilcoxon rank-sum
scipy.stats.ranksums(df.gtrends, np.random.normal(13, 12, 1000))



