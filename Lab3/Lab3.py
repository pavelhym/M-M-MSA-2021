import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from bayesian.train_bn import structure_learning, parameter_learning
from preprocess.discretization import get_nodes_type, discretization, code_categories, get_nodes_sign
from bayesian.save_bn import save_structure, save_params, read_structure, read_params
from bayesian.sampling import generate_synthetics
from external.libpgm.hybayesiannetwork import HyBayesianNetwork
from visualization.visualization import draw_BN
from bayesian.calculate_accuracy import calculate_acc

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

#Step 4
nodes_type = get_nodes_type(df)
print(nodes_type)

nodes_sign = get_nodes_sign(df)
print(nodes_sign)

discrete_data, coder = discretization(df, 'equal_frequency', ["CO(GT)", "PT08.S1(CO)", "C6H6(GT)", 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', "RH","AH"])

bayes_manual = dict(
    {'V': ["CO(GT)", "PT08.S1(CO)", "C6H6(GT)", 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
           'PT08.S5(O3)', 'T', "RH", "AH"],
     'E': [
         ['NOx(GT)', 'T'],
         ['NO2(GT)', 'T'],
         ['PT08.S4(NO2)', 'T'],

         ['CO(GT)', 'RH'],
         ['PT08.S1(CO)', 'RH'],
         ['NOx(GT)', 'RH'],
         ['T', 'RH'],

         ['NO2(GT)', 'AH'],
         ['PT08.S4(NO2)', 'AH'],
         ['NOx(GT)', 'AH'],
         ['T', 'AH'],
     ]
     }
)

print("Structure of bayesian network", bayes_manual)

draw_BN(bayes_manual, nodes_type, 'draw_bayes_manual')

# K2 algorithm
b_HC_K2 = structure_learning(discrete_data, 'HC', nodes_type, 'K2')
draw_BN(b_HC_K2, nodes_type, 'K2_bn')

# MI_algorithm
HC_mi = structure_learning(discrete_data, 'HC', nodes_type, 'MI')
draw_BN(HC_mi, nodes_type, 'MI_bn')

# Evo algor MI
mi_evo = structure_learning(discrete_data, 'evo', nodes_type, 'MI')

draw_BN(mi_evo, nodes_type, 'mi_evo')

params_manual = parameter_learning(df, nodes_type, bayes_manual, 'simple')
save_structure(bayes_manual, 'bayes_manual')
skel = read_structure('bayes_manual')
save_params(params_manual, 'bayes_params_manual')
params_manual = read_params('bayes_params_manual')
# combine skel(graph) and params(dist on v) -> learning bayes #4
bayes_manual = HyBayesianNetwork(skel, params_manual)

# generate syn data
syn_df_manual = generate_synthetics(bayes_manual, nodes_sign, 'simple', 800)

# plot targets
fix, axs = plt.subplots(3, 1, figsize=(20, 20))
sns.histplot(df['T'], ax=axs[0], kde=True)
sns.histplot(syn_df_manual['T'], ax=axs[0], color='purple', kde=True)
axs[0].legend(['Real data', 'Synthetic data'])
axs[0].set_title('T')

sns.histplot(df['AH'], ax=axs[1], kde=True)
sns.histplot(syn_df_manual['AH'], ax=axs[1], color='purple', kde=True)
axs[1].legend(['Real data', 'Synthetic data'])
axs[1].set_title('AH')

sns.histplot(df['RH'], ax=axs[2], kde=True)
sns.histplot(syn_df_manual['RH'], ax=axs[2], color='purple', kde=True)
axs[2].legend(['Real data', 'Synthetic data'])
axs[2].set_title('RH')
plt.show()

params = parameter_learning(df, nodes_type, b_HC_K2, 'simple')
save_structure(b_HC_K2, 'bayes_hc_k2')
skel = read_structure('bayes_hc_k2')
save_params(params, 'bayes_hc_k2_params')
params = read_params('bayes_hc_k2_params')
b_HC_K2 = HyBayesianNetwork(skel, params)

syn_df_k2 = generate_synthetics(b_HC_K2, nodes_sign, 'simple', 800)

figure, axs = plt.subplots(3, 1, figsize=(20, 20))

# targets
sns.histplot(df['T'], ax=axs[0], kde=True)
sns.histplot(syn_df_k2['T'], ax=axs[0], color='purple', kde=True)
axs[0].legend(['Real data', 'Synthetic data'])
axs[0].set_title('T')

sns.histplot(df['AH'], ax=axs[1], kde=True)
sns.histplot(syn_df_k2['AH'], ax=axs[1], color='purple', kde=True)
axs[1].legend(['Real data', 'Synthetic data'])
axs[1].set_title('AH')

sns.histplot(df['RH'], ax=axs[2], kde=True)
sns.histplot(syn_df_k2['RH'], ax=axs[2], color='purple', kde=True)
axs[2].legend(['Real data', 'Synthetic data'])
axs[2].set_title('RH')
plt.show()

params = parameter_learning(df, nodes_type, HC_mi, 'simple')
save_structure(HC_mi, 'bayes_hc_mi')
skel = read_structure('bayes_hc_mi')
save_params(params, 'bayes_hc_mi_params')
params = read_params('bayes_hc_mi_params')
HC_mi = HyBayesianNetwork(skel, params)

syn_df_mi = generate_synthetics(HC_mi, nodes_sign, 'simple', 800)

figure, axs = plt.subplots(3, 1, figsize=(20, 20))

# targets
sns.histplot(df['T'], ax=axs[0], kde=True)
sns.histplot(syn_df_mi['T'], ax=axs[0], color='purple', kde=True)
axs[0].legend(['Real data', 'Synthetic data'])
axs[0].set_title('T')

sns.histplot(df['AH'], ax=axs[1], kde=True)
sns.histplot(syn_df_mi['AH'], ax=axs[1], color='purple', kde=True)
axs[1].legend(['Real data', 'Synthetic data'])
axs[1].set_title('AH')

sns.histplot(df['RH'], ax=axs[2], kde=True)
sns.histplot(syn_df_mi['RH'], ax=axs[2], color='purple', kde=True)
axs[2].legend(['Real data', 'Synthetic data'])
axs[2].set_title('RH')
plt.show()

params_evo = parameter_learning(df, nodes_type, mi_evo, 'simple')
save_structure(mi_evo, 'bayes_evo_mi')
skel = read_structure('bayes_evo_mi')
save_params(params_evo, 'bayes_evo_mi_params')
params_evo = read_params('bayes_evo_mi_params')
mi_evo = HyBayesianNetwork(skel, params_evo)

syn_df_mi_evo = generate_synthetics(mi_evo, nodes_sign, 'simple', 800)

figure, axs = plt.subplots(3, 1, figsize=(20, 20))

# targets
sns.histplot(df['T'], ax=axs[0], kde=True)
sns.histplot(syn_df_mi_evo['T'], ax=axs[0], color='purple', kde=True)
axs[0].legend(['Real data', 'Synthetic data'])
axs[0].set_title('T')

sns.histplot(df['AH'], ax=axs[1], kde=True)
sns.histplot(syn_df_mi_evo['AH'], ax=axs[1], color='purple', kde=True)
axs[1].legend(['Real data', 'Synthetic data'])
axs[1].set_title('AH')

sns.histplot(df['RH'], ax=axs[2], kde=True)
sns.histplot(syn_df_mi_evo['RH'], ax=axs[2], color='purple', kde=True)
axs[2].legend(['Real data', 'Synthetic data'])
axs[2].set_title('RH')
plt.show()

accuracy_dict, rmse_dict, real_param, pred_param, indexes = calculate_acc(bayes_manual, df, ['T', "AH", "RH"], 'simple')
print("acc_manual")
print(accuracy_dict)
print("rmse_manual")
print(rmse_dict)
print("real_manual")
print(real_param)
print("predicted_manual")
print(pred_param)

accuracy_dict, rmse_dict, real_param, pred_param, indexes = calculate_acc(b_HC_K2, df, ['T', 'AH', 'RH'], 'simple')
print("acc_HC_K2")
print(accuracy_dict)
print("rmse_HC_K2")
print(rmse_dict)
print("real_HC_K2")
print(real_param)
print("predicted_HC_K2")
print(pred_param)

accuracy_dict, rmse_dict, real_param, pred_param, indexes = calculate_acc(HC_mi, df, ['T', 'AH', 'RH'], 'simple')
print("acc_HC_mi")
print(accuracy_dict)
print("rmse_HC_mi")
print(rmse_dict)
print("real_HC_mi")
print(real_param)
print("predicted_HC_mi")
print(pred_param)

accuracy_dict, rmse_dict, real_param, pred_param, indexes = calculate_acc(mi_evo, df, ['T', 'AH', 'RH'], 'simple')
print("acc_mi_evo")
print(accuracy_dict)
print("rmse_mi_evo")
print(rmse_dict)
print("real_mi_evo")
print(real_param)
print("predicted_mi_evo")
print(pred_param)
