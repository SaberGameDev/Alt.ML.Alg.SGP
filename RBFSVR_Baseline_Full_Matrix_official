###This exact code runs for predicting Mo. If this needs to be changed, then the index on line 14 will have to be changed###

#-------Modules from global Python and sci-kit learning----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error

df = pd.read_csv("Fe_3_traces_full_matrix.csv")
X = df.iloc[:,0:12]
y = df.iloc[:,12]

#It was found that using a manual library vs the LabelEncoder for transforming cat data to num data had insignificant differences in RMSE. LabelEncoder is used for simplicity
label_encoder = LabelEncoder()
x_cat = X.select_dtypes(include=['object']).apply(label_encoder.fit_transform) #This could also be done with the OneHotEncoder
x_num = X.select_dtypes(exclude=['object']).values
x = pd.concat([pd.DataFrame(x_num), x_cat], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

#There is a significant difference between RMSE values if data is Z-standardised. Both versions are on here.
svr = SVR(kernel='rbf', gamma=0.1, C=100, epsilon=0.1)
#------Non_scaled--------

instance_svr = svr.fit(X_train, y_train)
y_pred_train = svr.predict(X_train)
y_pred_test = svr.predict(X_test)

#-------Scaled--------

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scalar.transform(X_test)

#--------Plotting RMSE predictions of Train and Test----------
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Distribution of Predicted and Real Mo data with RBF')

ax1.plot(y_train, y_pred_train, '.')
ax1.set_ylabel('Predicted Training Data')
ax1.set_xlabel('Real Training Data')

lims_1 = [
  np.min([ax1.get_xlim(), ax1.get_ylim()]),
  np.max([ax1.get_xlim(), ax1.get_ylim()]),
]

ax1.text(0, 570, 'RMSE:' + str(root_mean_squared_error(y_train, y_pred_train)))

ax1.plot(lims_1, lims_1, 'k-', zorder=0)

ax2.plot(y_test, y_pred_test, '.')
ax2.set_ylabel('Predicted Testing Data')
ax2.set_xlabel('Real Testing Data')

lims_2 = [
    np.min([ax2.get_xlim(), ax2.get_ylim()]),
    np.max([ax2.get_xlim(), ax2.get_ylim()]),
]

ax2.text(0, 500, 'RMSE:' + str(root_mean_squared_error(y_test, y_pred_test)))

ax2.plot(lims_2, lims_2, 'k-', zorder=0)

plt.show()
