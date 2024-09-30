#Random Forest is an ensemble method; ensemble methods combine predictions from
#several base estimators built with a given learning algorithm to improve
#generalizability/robustness over a single estimator

#*Z-standardization of the data for RF does not significantly impact RMSE when compared to non-scaled data

#-------Modules from global Python and sci-kit learn--------
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

#------Data set-up------
df = pd.read_csv("Fe_3_traces_full_matrix.csv")

X = df.iloc[:,0:12]
X['RANDOM'] = np.random.RandomState(42).randn(X.shape[0])
y = df.iloc[:,12] #Mo

label_encoder = LabelEncoder()
x_cat = X.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
x_num = X.select_dtypes(exclude=['object']).values
x = pd.concat([pd.DataFrame(x_num), x_cat], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

#-----Regression------

regr = RandomForestRegressor(max_features=3, n_estimators=500, random_state=0)
regr.fit(X_train, y_train)

y_pred_train = regr.predict(X_train)
y_pred_test = regr.predict(X_test)

#-----RMSE Plotting-------

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Distribution of Predicted and Real Mo data with RF')

ax1.plot(y_train, y_pred_train, '.')
ax1.set_ylabel('Predicted Training Data (Mo [ppm])')
ax1.set_xlabel('Real Training Data (Mo [ppm])')

lims_1 = [
    np.min([ax1.get_xlim(), ax1.get_ylim()]),
    np.max([ax1.get_xlim(), ax1.get_ylim()]),
]

ax1.text(0, 570, 'RMSE:' + str(root_mean_squared_error(y_train, y_pred_train)))

ax1.plot(lims_1, lims_1, 'k-', zorder=0)

ax2.plot(y_test, y_pred_test, '.')
ax2.set_ylabel('Predicted Testing Data (Mo [ppm])')
ax2.set_xlabel('Real Testing Data (Mo [ppm])')

lims_2 = [
    np.min([ax2.get_xlim(), ax2.get_ylim()]),
    np.max([ax2.get_xlim(), ax2.get_ylim()]),
]

ax2.text(0, 500, 'RMSE:' + str(root_mean_squared_error(y_test, y_pred_test)))

ax2.plot(lims_2, lims_2, 'k-', zorder=0)

plt.show()

#-------Decrease in Node Impurity (Var Importance 1)--------
importances = pd.Series(regr.feature_importances_, index=X.columns)
importances.sort_values(ascending=True, inplace=True)
importances.plot.barh(color="green")
plt.xlabel("Mean Decrease in Node Impurity")
plt.title("Built in Method (MDI) for Feature Importance")

plt.show()

#--------Feature Permutation [good for high-cardinality features] (Var Importance 2)-------
result = permutation_importance(regr, X_test, y_test, n_repeats=10, random_state=42)
importances = pd.Series(result.importances_mean, index=X.columns)
importances.sort_values(ascending=True, inplace=True)
importances.plot.barh(color="green")
plt.xlabel("Mean accuracy decrease")
plt.title("Permutaion for Feature Importance")

plt.show()

