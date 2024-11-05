import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt


#--------------Data Setup (Mo)--------------
df = pd.read_csv("Fe_3_traces_full_matrix_v2.csv")
mo_df = df[(df['FeHR/FeT'] >= 0.38) & (df['Fe_py/FeHR'] >= 0.7)]
mo_df = mo_df.loc[:, ~mo_df.columns.isin(['v_ppm','u_ppm','u_prep_methods',
                                        'u_exp_methods','u_ana_methods',
                                        'v_prep_methods','v_exp_methods','v_ana_methods'])]
mo_df = mo_df.reset_index(drop=True)


#---------------Categorical Data Conversion--------------
#Check and handle cat vars (OneHotEncoder)
cat_columns = mo_df.select_dtypes(include=['object']).columns.tolist()
hot_boi = OneHotEncoder(sparse_output=False)

one_hot_encoded = hot_boi.fit_transform(mo_df[cat_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=hot_boi.get_feature_names_out(cat_columns))

mo_df_encoded = pd.concat([mo_df, one_hot_df], axis=1)
mo_df_encoded = mo_df_encoded.drop(cat_columns, axis=1)

x = mo_df_encoded.loc[:, ~mo_df_encoded.columns.isin(['mo_ppm'])]
y = mo_df_encoded.loc[:, mo_df_encoded.columns == 'mo_ppm']


#----------------Prediction and RMSE Value------------------

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

##scaler = StandardScaler().fit(X_train)
##X_train_scaled = scaler.transform(X_train)
##X_test_scaled = scaler.transform(X_test)

regr = RandomForestRegressor(max_features=3, n_estimators=500, random_state=0)
#Rich has 3 features in his trees, and it looks like 500 trees per forest

regr.fit(X_train, y_train)

y_pred_train = regr.predict(X_train)
y_pred_test = regr.predict(X_test)


#--------- Plotting RMSE for both y_train and y_test-------

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Distribution of Predicted and Real Mo data with RF')

ax1.plot(y_train, y_pred_train, '.')
ax1.set_ylabel('Predicted Training Data (Mo [ppm])')
ax1.set_xlabel('Real Training Data (Mo [ppm])')

lims_1 = [
    np.min([ax1.get_xlim(), ax1.get_ylim()]),
    np.max([ax1.get_xlim(), ax1.get_ylim()]),
]

ax1.annotate('RMSE:' + str(root_mean_squared_error(y_train, y_pred_train)), xy=(0.05, 0.95), xycoords='axes fraction')

ax1.plot(lims_1, lims_1, 'k-', zorder=0)

ax2.plot(y_test, y_pred_test, '.')
ax2.set_ylabel('Predicted Testing Data (Mo [ppm])')
ax2.set_xlabel('Real Testing Data (Mo [ppm])')

lims_2 = [
    np.min([ax2.get_xlim(), ax2.get_ylim()]),
    np.max([ax2.get_xlim(), ax2.get_ylim()]),
]

ax2.annotate('RMSE:' + str(root_mean_squared_error(y_test, y_pred_test)), xy=(0.05, 0.95), xycoords='axes fraction')

ax2.plot(lims_2, lims_2, 'k-', zorder=0)

plt.show()

#---------------Iterating for Feature Importance---------------

#Number of iterations (100) + Initialize all results
##n = 100
##all_results = []
##
##for i in range(n):
##    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
##    regr.fit(X_train, y_train)
##    y_pred_train = regr.predict(X_train)
##    y_pred_test = regr.predict(X_test)
##    importances = pd.Series(regr.feature_importances_, index=x.columns)
##    all_results.append(importances)
##
##results_df = pd.DataFrame(all_results)
##
##plt.boxplot(results_df, vert=False, tick_labels=x.columns)
##plt.ylabel('Feature')
##plt.xlabel('Mean Decrease in Node Impurity')
##
##plt.show()
