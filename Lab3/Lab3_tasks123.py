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
from scipy.stats import chi2
from scipy.stats import lognorm
import scipy
from scipy import stats
import statsmodels.nonparametric.kernel_density
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pylab
from sklearn.linear_model import LassoLarsIC


df = pd.read_excel('./mydata/AirQualityUCI.xlsx')

for i in df.columns[2:]:
    df[str(i)].replace({-200: None}, inplace=True)
    mean = np.mean(df[str(i)])
    df[str(i)].replace({None: mean}, inplace=True)
    df[str(i)] = pd.to_numeric(df[str(i)])

df = df.groupby(['Date']).agg({"CO(GT)": "mean",
                               "PT08.S1(CO)": "mean",
                               "C6H6(GT)": "mean",
                               'PT08.S2(NMHC)': 'mean',
                               'NOx(GT)': 'mean',
                               'PT08.S3(NOx)': 'mean',
                               'NO2(GT)': 'mean',
                               'PT08.S4(NO2)': 'mean',
                               'PT08.S5(O3)': 'mean',
                               'T': "mean",
                               "RH": 'mean',
                               "AH": "mean"
                               }).reset_index()

date = df['Date']

df = df[["CO(GT)", "PT08.S1(CO)", "C6H6(GT)", 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
         'PT08.S5(O3)', 'T', "RH", "AH"]]


def get_best_distribution(data):
    dist_names = ["norm", "expon", 'chi2', "exponnorm", "lognorm", "gamma"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)

        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    return best_dist, best_p, params[best_dist]


# Useful functions from prev labs
# dict with fits of distributions
dist_name_to_func = {
    "norm": (lambda column: scipy.stats.norm.fit(column)),
    "exponnorm": (lambda column: scipy.stats.exponnorm.fit(column)),
    "genextreme": (lambda column: scipy.stats.genextreme.fit(column)),
    "expon": (lambda column: scipy.stats.expon.fit(column)),
    "chi2": (lambda column: scipy.stats.chi2.fit(column)),
    "lognorm": (lambda column: scipy.stats.lognorm.fit(column)),
    "gamma": (lambda column: scipy.stats.gamma.fit(column)),
    "exponweib": (lambda column: scipy.stats.exponweib.fit(column)),
    "weibull_max": (lambda column: scipy.stats.weibull_max.fit(column)),
    "weibull_min": (lambda column: scipy.stats.weibull_min.fit(column)),
    "pareto": (lambda column: scipy.stats.pareto.fit(column))

}


# return parameters
def parameters(column, dist):
    return dist_name_to_func[dist](column)


# Step 1

target = df[["T", 'RH', 'AH']]

predictors = df.drop(columns=["T", 'RH', 'AH'])


# Step 2
# inverse transform sampling

def inverse_transform_sampling(n, distribution, params):
    uniform_dist = np.random.uniform(size=n)
    gd = distribution(*params)
    return gd.ppf(uniform_dist)


param = []
for i in target:
    params = parameters(df[str(i)], get_best_distribution(df[str(i)])[0])
    param.append(params)
#print(param)

sample_T = inverse_transform_sampling(10000, chi2, param[0])
sample_RH = inverse_transform_sampling(10000, gamma, param[1])
sample_AH = inverse_transform_sampling(10000, lognorm, param[2])
fig, axs = plt.subplots(1, 3, figsize=(20, 20))
sns.distplot(df['T'], kde=True, label='initial data', ax=axs[0])
sns.distplot(sample_T, kde=True, norm_hist=True, label='sampled data', ax=axs[0])

sns.distplot(df['AH'], kde=True, norm_hist=True, label='initial data', ax=axs[1])
sns.distplot(sample_AH, kde=True, norm_hist=True, label='sampled data', ax=axs[1])

sns.distplot(df['RH'], kde=True, norm_hist=True, label='initial data', ax=axs[2])
sns.distplot(sample_RH, kde=True, norm_hist=True, label='sampled data', ax=axs[2])
plt.show()
# rejection sampling

# make some dict with pdf
dist_name_to_func_pdf = {
    "norm": (lambda i, j: scipy.stats.norm.pdf(i, *j)),
    "exponnorm": (lambda i, j: scipy.stats.exponnorm.pdf(i, *j)),
    "genextreme": (lambda i, j: scipy.stats.genextreme.pdf(i, *j)),
    "expon": (lambda i, j: scipy.stats.expon.pdf(i, *j)),
    "chi2": (lambda i, j: scipy.stats.chi2.pdf(i, *j)),
    "lognorm": (lambda i, j: scipy.stats.lognorm.pdf(i, *j)),
    "gamma": (lambda i, j: scipy.stats.gamma.pdf(i, *j)),
    "exponweib": (lambda i, j: scipy.stats.exponweib.pdf(i, *j)),
    "weibull_max": (lambda i, j: scipy.stats.weibull_max.pdf(i, *j)),
    "weibull_min": (lambda i, j: scipy.stats.weibull_min.pdf(i, *j)),
    "pareto": (lambda i, j: scipy.stats.pareto.pdf(i, *j))

}


def brute_forse(func, a, b, eps):
    bigest_so_far = a
    max_func_value = func(a)
    n = math.ceil((b - a) / eps)
    for k in range(1, n + 1):
        x = a + k * (b - a) / n
        fx = func(x)
        if fx > max_func_value:
            bigest_so_far = x
            max_func_value = fx
    return max_func_value


def rejection_samp(p, q, q_sample, max_x=800000, iterations=10 ** 5):
    number_of_steps = 100000
    # M = brute_forse((lambda x: p(x)/q(x)), 0, max_x, max_x/number_of_steps)
    M = 5
    # print(M)
    # collect all accepted samples here
    samples = []

    # try this many candidates
    N = iterations

    for _ in range(N):
        # sample a candidate
        candidate = q_sample()

        # calculate probability of accepting this candidate
        prob_accept = p(candidate) / (M * q(candidate))

        # accept with the calculated probability
        if np.random.random() < prob_accept:
            samples.append(candidate)
    return samples, M


M = 5

x = df['T']
density = kde.gaussian_kde(x)  # p function(for rejection sampling)
app_name, app_pvalue, app_params = get_best_distribution(df['T'])
app_density = lambda y: dist_name_to_func_pdf[app_name](y, app_params)  # q
samples = rejection_samp(density, app_density,
                         lambda: scipy.stats.chi2.rvs(*app_params))  # we use non-comon dist here!!!
xgrid = np.linspace(x.min(), x.max(), 100)
# plt.hist(x, bins=8,density=True, stacked=True)
#print(samples)
plt.hist(samples, bins=1000, density=True, stacked=True)
plt.plot(xgrid, density(xgrid), 'r-')
plt.plot(xgrid, M * app_density(xgrid), 'g-')
plt.title('T' + " " + "histogram")
plt.show()

x = df['RH']
density = kde.gaussian_kde(x)  # p function(for rejection sampling)
app_name, app_pvalue, app_params = get_best_distribution(df['RH'])
app_density = lambda y: dist_name_to_func_pdf[app_name](y, app_params)  # q
samples = rejection_samp(density, app_density,
                         lambda: scipy.stats.gamma.rvs(*app_params))  # we use non-comon dist here!!!
xgrid = np.linspace(x.min(), x.max(), 100)
# plt.hist(x, bins=8,density=True, stacked=True)
#print(samples)
plt.hist(samples, bins=1000, density=True, stacked=True)
plt.plot(xgrid, density(xgrid), 'r-')
plt.plot(xgrid, M * app_density(xgrid), 'g-')
plt.title('RH' + " " + "histogram")
plt.show()

x = df['AH']
density = kde.gaussian_kde(x)  # p function(for rejection sampling)
app_name, app_pvalue, app_params = get_best_distribution(df['AH'])
app_density = lambda y: dist_name_to_func_pdf[app_name](y, app_params)  # q
samples = rejection_samp(density, app_density,
                         lambda: scipy.stats.lognorm.rvs(*app_params))  # we use non-comon dist here!!!
xgrid = np.linspace(x.min(), x.max(), 100)
# plt.hist(x, bins=8,density=True, stacked=True)
#print(samples)
plt.hist(samples, bins=1000, density=True, stacked=True)
plt.plot(xgrid, density(xgrid), 'r-')
plt.plot(xgrid, M * app_density(xgrid), 'g-')
plt.title('AH' + " " + "histogram")
plt.show()

# Step 3
# Building a correlation matrix of features
df.iloc[:]
corr = df.iloc[:].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', ax=ax, cmap='Blues')
plt.show()

f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr()[[ "AH", 'RH','T']], vmin = -1, vmax = +1, annot = True, cmap = 'Blues')
plt.show()
