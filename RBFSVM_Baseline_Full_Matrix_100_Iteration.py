import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

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


#~*~*~*~*~*~*~*~*~*~*~*~*~100 Iterations~*~*~*~*~*~*~*~*~*~*~*~*~

prediction_array = [0] * 100
mse_array = [0] * 100

svr = SVR(kernel='rbf', gamma=0.1, C=100, epsilon=0.1)



for i in range(100):
    train, test = train_test_split(df, test_size = 0.2)

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
    instance_svr = svr.fit(X_train_scaled, y_train)
    prediction_array[i] = svr.predict(X_train_scaled)
    mse_array[i] = mean_squared_error(y_train.values, prediction_array[i])

    plt.plot(train['interpreted_age'], prediction_array[i], color='red', alpha=0.1)


#~*~*~*~*~*~*~*~*~*~*~*~*~MSE Boxplot~*~*~*~*~*~*~*~*~*~*~*~*~
plt.gca().invert_xaxis()
plt.show()
