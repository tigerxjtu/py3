import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


df = pd.read_csv(r'C:\projects\python\data\dataguru\dl\train.csv')
train_cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']

X=df[train_cols]
y=df['SalePrice']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


xgb1 = xgb.XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=500, silent=False, objective='reg:gamma')
model = xgb1.fit(X_train, y_train)

# Predict training set:
dtrain_predictions = xgb1.predict(X_train)

# Print model report:
print("\nModel Report on Train")
mse = mean_squared_error(y_train.values, dtrain_predictions)
print(f'mean square error: {mse}')
score = xgb1.score(X_train, y_train)
print(f'score is {score}')

print("\nModel Report on Test")
mse = mean_squared_error(y_test.values, xgb1.predict(X_test))
print(f'mean square error: {mse}')
score = xgb1.score(X_test, y_test)
print(f'score is {score}')

plot_importance(model)
plt.show()
