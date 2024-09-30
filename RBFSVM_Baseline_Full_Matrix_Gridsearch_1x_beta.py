#-----Visual Grid Search for best RBF parameters (beta)-----

#-----Modules from global Python and sci-kit learning------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error
import seaborn as sns

df = pd.read_csv("Fe_3_traces_full_matrix.csv")

X = df.iloc[:,0:12]
y = df.iloc[:,12]

label_encoder = LabelEncoder()
x_categorical = X.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
x_numerical = X.select_dtypes(exclude=['object']).values
x = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

C_range = [0.1, 1, 10, 100, 1000]
gamma_range= [1, 0.1, 0.01, 0.001, 0.0001]

df_rmse_train = pd.DataFrame(index=C_range, columns=gamma_range)
df_rmse_test = pd.DataFrame(index=C_range, columns=gamma_range)

#For non_scaled grid search. Comment out one or the other
for C_val in C_range:
    for gamma_val in gamma_range:
        svr = SVR(kernel='rbf', gamma=gamma_val, C=C_val, epsilon=0.1)
        instance_svr = svr.fit(X_train, y_train)
        y_pred_train = svr.predict(X_train)
        y_pred_test = svr.predict(X_test)
        df_rmse_train.at[C_val, gamma_val] = root_mean_squared_error(y_train, y_pred_train)
        df_rmse_test.at[C_val, gamma_val] = root_mean_squared_error(y_test, y_pred_test)

#For scaled grid search. Comment out one or the other
for C_val in C_range:
    for gamma_val in gamma_range:
        svr = SVR(kernel='rbf', gamma=gamma_val, C=C_val, epsilon=0.1)
        instance_svr = svr.fit(X_train_scaled, y_train)
        y_pred_train = svr.predict(X_train_scaled)
        y_pred_test = svr.predict(X_test_scaled)
        df_rmse_train.at[C_val, gamma_val] = root_mean_squared_error(y_train, y_pred_train)
        df_rmse_test.at[C_val, gamma_val] = root_mean_squared_error(y_test, y_pred_test)

df_rmse_train = df_rmse_train[df_rmse_train.columns].astype(float)
df_rmse_test = df_rmse_test[df_rmse_test.columns].astype(float)

plt.figure(1)
sns.heatmap(df_rmse_train, annot=True)
plt.figure(2)
sns.heatmap(df_rmse_test, annot=True)

plt.show()

