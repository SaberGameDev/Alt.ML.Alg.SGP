import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

#Notes for things to address later:
    #1. scikit-learn can imputate missing values with either univariate or multivariate (probably this)

#Step 1: Data Preprocessing. This run is using a full-matrix filtered from the dataset
    #where all samples have to have at least some value in the 12 explanatory categories AND
    # a value for Mo, U, and V.

    #Because this filtering step has been done, I don't think there is a need to correct for imbalanced
    #datasets

    #I'm changing non-numerical entries into numerical using a dictionary and mapping them to each category.

df = pd.read_csv("Fe_3_traces_full_matrix.csv")
pd.set_option('display.max_columns', None)
#site_type
d = {'core' : 0, 'cuttings' : 1, 'outcrop': 2}
df['site_type'] = df['site_type'].map(d)
#basin_type
d = {'back-arc' : 0, 'fore-arc' : 1, 'foreland - peripheral' : 2,
     'foreland - retro-arc' : 3, 'intracratonic sag' : 4, 'passive margin' : 5,
     'rift' : 6, 'wrench' : 7}
df['basin_type'] = df['basin_type'].map(d)
#meta_bin
d = {'anchizone' : 0, 'diagenetic' : 1, 'epizone' : 2}
df['meta_bin'] = df['meta_bin'].map(d)
#lith_name
d = {'anhydrite' : 0, 'carbonate' : 1, 'chert' : 2, 'crystalline limestone' : 3,
     'dolomite' : 4, 'dolomudstone' : 5, 'iron formation' : 6, 'lime mudstone' : 7,
     'limestone' : 8, 'marl' : 9, 'mudstone' : 10, 'ore' : 11, 'phosphorite' : 12,
     'sandstone' : 13, 'shale' : 14, 'siltstone' : 15}
df['lith_name'] = df['lith_name'].map(d)
#dep_env_bin
d = {'basinal (marine)' : 0, 'inner shelf (marine)' : 1, 'lacustrine' : 2,
     'outer shelf (marine)' : 3}
df['dep_env_bin'] = df['dep_env_bin'].map(d)


#Change the categorical to numerical system without considering the data to be continuous
#df.corr() to see what fits best

df_continuous = df[['FeHR/FeT', 'Fe_py/FeHR', 'TOC_wt_per', 'Al _wt_per',
        'Mo_ppm', 'U_ppm', 'V_ppm']]

#For Mo_ppm
df_sorted = df.sort_values('interpreted_age')

X = df_sorted[['interpreted_age', 'lat_dec', 'long_dec', 'site_type', 'basin_type',
        'meta_bin', 'lith_name', 'dep_env_bin', 'FeHR/FeT', 'Fe_py/FeHR',
        'TOC_wt_per', 'Al _wt_per']].to_numpy()
y = np.array(df_sorted['Mo_ppm'].values.tolist())

train, test = train_test_split(df, test_size = 0.2, random_state=42)

train = train.sort_values('interpreted_age')
test = test.sort_values('interpreted_age')

X_train, X_test = train[['interpreted_age', 'lat_dec', 'long_dec', 'site_type',
                         'basin_type', 'meta_bin', 'lith_name', 'dep_env_bin',
                         'FeHR/FeT', 'Fe_py/FeHR', 'TOC_wt_per', 'Al _wt_per']], test[['interpreted_age', 'lat_dec', 'long_dec', 'site_type',
                         'basin_type', 'meta_bin', 'lith_name', 'dep_env_bin',
                         'FeHR/FeT', 'Fe_py/FeHR', 'TOC_wt_per', 'Al _wt_per']]
y_train, y_test = train['Mo_ppm'], test['Mo_ppm']

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr = SVR(kernel='rbf', gamma=0.1, C=100, epsilon=0.1)
#should epsilon be the mean squared error?
instance_svr = svr.fit(X_train_scaled, y_train)
y_pred_train = svr.predict(X_train_scaled)

y_pred_test = svr.predict(X_test_scaled)


#Stockey does 100 random forest iterations. So if it's not too computationally taxing, we should do that too.

plt.scatter(train['interpreted_age'], train['Mo_ppm'], label="Mo data",facecolor="blue",
           edgecolor="blue", s=15)
plt.plot(train['interpreted_age'], y_pred_train,
        label="model", color="red")
plt.gca().invert_xaxis()
plt.xlabel('Time (Ma)')
plt.ylabel('Mo (ppm)')
#no partial dependence shading yet. How many iterations is good to run?
plt.show()

#RMSE instead of MSE
#Grid Search
