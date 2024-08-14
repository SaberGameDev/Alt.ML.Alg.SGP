import math
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler #PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV #KFold, cross_val_score  StratifiedShuffleSplit
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

#Notes for things to address later:
    #1. scikit-learn can imputate missing values with either univariate or multivariate (probably this)

#Step 1: Data Preprocessing. This run is using a full-matrix filtered from the dataset
    #where all samples have to have at least some value in the 12 explanatory categories AND
    # a value for Mo, U, and V.

    #Because this filtering step has been done, I don't think there is a need to correct for imbalanced
    #datasets

    #I'm changing non-numerical entries into numerical using a dictionary and mapping them to each category.

df = pd.read_csv("Fe_3_traces_full_matrix.csv")
###site_type
##d = {'core' : 0, 'cuttings' : 1, 'outcrop': 2}
##df['site_type'] = df['site_type'].map(d)
###basin_type
##d = {'back-arc' : 0, 'fore-arc' : 1, 'foreland - peripheral' : 2,
##     'foreland - retro-arc' : 3, 'intracratonic sag' : 4, 'passive margin' : 5,
##     'rift' : 6, 'wrench' : 7}
##df['basin_type'] = df['basin_type'].map(d)
###meta_bin
##d = {'anchizone' : 0, 'diagenetic' : 1, 'epizone' : 2}
##df['meta_bin'] = df['meta_bin'].map(d)
###lith_name
##d = {'anhydrite' : 0, 'carbonate' : 1, 'chert' : 2, 'crystalline limestone' : 3,
##     'dolomite' : 4, 'dolomudstone' : 5, 'iron formation' : 6, 'lime mudstone' : 7,
##     'limestone' : 8, 'marl' : 9, 'mudstone' : 10, 'ore' : 11, 'phosphorite' : 12,
##     'sandstone' : 13, 'shale' : 14, 'siltstone' : 15}
##df['lith_name'] = df['lith_name'].map(d)
###dep_env_bin
##d = {'basinal (marine)' : 0, 'inner shelf (marine)' : 1, 'lacustrine' : 2,
##     'outer shelf (marine)' : 3}
##df['dep_env_bin'] = df['dep_env_bin'].map(d)


#Scaling (Z-score)


X = df[['FeHR/FeT', 'Fe_py/FeHR', 'TOC_wt_per', 'Al _wt_per',
        'Mo_ppm', 'U_ppm', 'V_ppm']]

scaledX = StandardScaler().fit_transform(X)

df1 = pd.DataFrame(scaledX)

##df['FeHR/FeT'] = df1[0].values
##df['Fe_py/FeHR'] = df1[1].values
##df['TOC_wt_per'] = df1[2].values
##df['Al _wt_per'] = df1[3].values
##df['Mo_ppm'] = df1[4].values
##df['U_ppm'] = df1[5].values
##df['V_ppm'] = df1[6].values
#df.to_csv("full_matrix_num.csv")

X = df.iloc[:,0:1] #'FeHR/FeT'], 'Fe_py/FeHR', 'TOC_wt_per', 'Al _wt_per']]
#Testing for Mo prediction
y = df['Mo_ppm']

svr = SVR(kernel='rbf', C=1, gamma='auto', epsilon=0.1)
svr.fit(X, y)

y_pred = svr.predict(X)

plt.scatter(X, y, label='data')
plt.plot(X, y_pred, color='darkorange', label='prediction')
plt.xlabel("FeHR/FeT")
plt.ylabel("Mo (ppm)")
plt.legend()
plt.show()
#~*~*~*~*~*~*~*~Train-Test Continuous~*~*~*~*~*~*~*~
#parameter grid
##param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
##              'gamma':[0.0001, 0.001, 0.01, 1, 10, 100, 1000]}
##
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#~*~*~*~*~*~*~*~Model Training~*~*~*~*~*~*~*~

##for i, C in enumerate(param_grid['C']):
##    for j, gamma in enumerate(param_grid['gamma']):
##        clf = SVC(kernel='rbf', C=C, gamma=gamma)
##        clf.fit(X_train, y_train)
##        y_pred = clf.predict(X_test)
##        accuracy = accuracy_score(y_test, y_pred)
##        axs[i,j].scatter(X_test[:,0], X_test[:,1], c=y_pred)
##        axs[i,j].set_xticks(())
##        axs[i,j].set_yticks(())
##
##axs[i,j].set_title('C = {}, gamma = {}\nAccuracy = {:.2f}'.format(
##            C, gamma, accuracy))
## 
##plt.show()

#~*~*~*~*~*~*~*~Transformations and Such~*~*~*~*~*~*~*~

#With histogram bin = 100:
    #Mo_ppm is right skewed
    #U_ppm is right skewed
    #V_ppm is right skewed
    #Al wt% is kind of normal/bimodal
    #TOC wt% is right skewed
    #Fe_py/FeHR is a combination of high right skewed, normal, and lower left skewed
    #FeHR/FeT is pretty normal looking

#Scaler options to considers:
    #RobustScaler
    #PowerTransformer (Box-Cox)
        #Makes data more Gaussian
        #For strictly positive values
    #QuantileTransformer (Gaussian output)
        #Seems good for multivariate
        #Outlier collapse can cause saturation artifacts for extreme values

#Can't use box-cox because there are 0 values. Only works with strictly positive
##pt = PowerTransformer(method='yeo-johnson', standardize=True)
##data = [df['FeHR/FeT'], df['Fe_py/FeHR'], df['TOC_wt_per'], df['Al _wt_per'],
##        df['Mo_ppm'], df['U_ppm'], df['V_ppm']]
##
##geochem_transformed = pt.fit_transform(data)
##
#I still need to automate the process of annealing the transformed geochem dataset to the
#original dataset. Until them, I manually copy-and-pasted it into a new form

#~*~*~*~*~*~*~*~Hyperparameters~*~*~*~*~*~*~*~
#Definition: A parameter which specifies details of learning process (e.g., learning rate or optimiser
#choice

#Cross validation

#I should grab the data that is missing trace metal data to be the external tester.
