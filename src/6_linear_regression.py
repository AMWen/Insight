### Linear regression

# Import necessary packages
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Regression packages
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

# Statistics packages
import statsmodels.api as sm
import statsmodels.stats.api as sms

# Load data
hospital_info = pd.read_csv("data/all_hospitals.csv")

# Create arrays for features and target variable
y = hospital_info['Average Rating'].values
all_features = hospital_info.drop(["Hospital Name", "Link", "State", "Address", "Summary"], axis=1)
X = all_features.drop(["Average Rating"], axis=1)

# Reshape y
y = y.reshape(-1, 1)

# Check the dimensions of X and y
print("Dimensions of y: {}".format(y.shape))
print("Dimensions of X: {}".format(X.shape))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=15)

## Linear Regression
# Create the regressor
lin_reg = LinearRegression()

# Fit the model
lin_reg.fit(X_train, y_train)

# Make predictions
y_pred = lin_reg.predict(X_test)

# Print out accuracies
test_rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
test_r2 = r2_score(y_test, y_pred)
print(test_rmse)
print(test_r2)

# Training error
y_train_pred = lin_reg.predict(X_train)

train_rmse = (np.sqrt(mean_squared_error(y_train, y_train_pred)))
train_r2 = r2_score(y_train, y_train_pred)
print(train_rmse)
print(train_r2)

# Pairwise scatter plot
sns.set(rc={'figure.figsize':(25,25)})
g = sns.pairplot(data=all_features)
