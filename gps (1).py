# -*- coding: utf-8 -*-
"""GPS.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mVmx9_TvRdpSo7HWfLVr5ufSjiIL6HPk
"""

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

from google.colab import drive
drive.mount('/content/drive')

# Data prepocessing
# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/GpaPredict/STUDENT_data.csv')

data.head()

data.tail()

#renaming the columns
data = data.rename(columns={'Level of interest in your course of study?': 'Course_LOI',
                       'Is learning environment conducive?': 'gudLearning_env',
                       'Jamb Score': 'Jamb',
                       'Do you have any health challenge?': 'Any_health_challenge',
                       'How often do you study?': 'Study_Rate',
                       'Are u fully sponsored': 'Sponsored',
                       'No of Friends': 'No_of_Friends',
                       'Level of Assimilation': 'Assimilation_rate',
                       'How often do you engage in Extra curricular activities?': 'Extra_activities',
                       'Are you in a Relationship?': 'inRelationship',
                       'How often do you attend classes?': 'Attendance_Rate',
                       'How many times do you eat in a day?': 'Rate_of_feeding',
                       'Employment Status': 'Employment_Status'})

data.head()

data.drop(data.columns[0], axis=1, inplace=True)

# EDA
print("Dataset shape: ", data.shape)
print("Dataset description: \n", data.describe())
print("Missing data: \n", data.isnull().sum())

data = data.dropna()

data.dtypes

# Fill in missing data
data = data.fillna(data.select_dtypes(include=np.number).mean())

data.isnull().sum()

data = data.dropna()

# Convert the Jamb column to numeric
data['Jamb'] = pd.to_numeric(data['Jamb'], errors='coerce')
# Drop any rows with NaN values
data.dropna(inplace=True)

data

data.columns

# Label encode categorical features
# create a LabelEncoder object
le = LabelEncoder()

# define the columns to be encoded
cols_to_le_encode = ['gudLearning_env', 'Any_health_challenge','Sponsored', 'Any_health_challenge', 'inRelationship',
                    'Employment_Status', 'No_of_Friends', 'Gender']
# encode the columns
for col in cols_to_le_encode:
    data[col] = le.fit_transform(data[col])

data.head()

# data.dropna(subset=['Rate_of_feeding'], inplace=True)

data

df_encoded = data

df_encoded.dtypes

df_encoded.dropna(inplace=True)

df_encoded.isnull().sum()

df_encoded

# convert GPA column to numeric, ignoring errors
df_encoded['GPA'] = pd.to_numeric(df_encoded['GPA'], errors='coerce')

# remove rows with NaN values
df_encoded.dropna(inplace=True)

#EDA
# Correlation matrix
corr_matrix = df_encoded.corr()
# Set the figure size
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='inferno')
plt.title("Correlation Matrix")
plt.show()

df_encoded.columns

df_encoded.drop(['No_of_Friends', 'Any_health_challenge'], axis = 1, inplace = True)

df_encoded.drop(['gudLearning_env', 'Gender'], axis = 1, inplace = True)

#EDA AFTER DROPPING SOME VARIABLES
# Correlation matrix
corr_matrix = df_encoded.corr()
# Set the figure size
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='inferno')
plt.title("Correlation Matrix")
plt.show()

#SAVING THE NEW DATASET INTO MY SYSTEM
df_encoded.to_csv('cleaned_dataset.csv', index=False)

# Load the clean dataset dataset
data1 = pd.read_csv('cleaned_dataset.csv')

data1.head()

data1.tail()

data1.columns

# Split data into train and test sets
# Split data into train and test sets
#X = df_encoded.drop(['GPA','Sponsored','Extra_activities','inRelationship','Rate_of_feeding','Employment_Status'], axis = 1)
X = data1[['Course_LOI', 'Rate_of_feeding', 'Jamb', 'Study_Rate', 'Assimilation_rate','Attendance_Rate', 'Extra_activities']]
y = data1['GPA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Verify the shape of the datasets
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

X.head()

X.tail()

# Decision Tree
det_model = DecisionTreeRegressor(random_state=42)
det_model.fit(X_train, y_train)
det_y_pred = det_model.predict(X_test)
det_mse = mean_squared_error(y_test, det_y_pred)
det_r2 = r2_score(y_test, det_y_pred)
det_mae = mean_absolute_error(y_test, det_y_pred)
det_maep = mean_absolute_percentage_error(y_test, det_y_pred)
print("Decision Tree Mean squared error: ", det_mse)
print("Decision Tree R-squared: ", det_r2)
print("Decision Tree Mean absolute error: ", det_mae)
print("Decision Tree Mean absolute % error: ", det_maep)

# Random Forest
raf_model = RandomForestRegressor(random_state=42)
raf_model.fit(X_train, y_train)
raf_y_pred = raf_model.predict(X_test)
raf_mse = mean_squared_error(y_test, raf_y_pred)
raf_r2 = r2_score(y_test, raf_y_pred)
raf_mae = mean_absolute_error(y_test, raf_y_pred)
raf_maep = mean_absolute_percentage_error(y_test, raf_y_pred)
print("Random Forest Mean squared error: ", raf_mse)
print("Random Forest R-squared: ", raf_r2)
print("Random Forest Mean absolute error: ", raf_mae)
print("Random Forest Mean absolut Percentage error: ", raf_maep)

# SVM
svm_model = SVR()
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)
svm_mse = mean_squared_error(y_test, svm_y_pred)
svm_r2 = r2_score(y_test, svm_y_pred)
svm_mae = mean_absolute_error(y_test, svm_y_pred)
svm_maep = mean_absolute_percentage_error(y_test, svm_y_pred)
print("SVM Mean squared error: ", svm_mse)
print("SVM R-squared: ", svm_r2)
print("SVM Mean absolute error: ", svm_mae)
print("SVM Mean absolute % error: ", svm_maep)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_y_pred)
lr_r2 = r2_score(y_test, lr_y_pred)
lr_mae = mean_absolute_error(y_test, lr_y_pred)
lr_maep = mean_absolute_percentage_error(y_test, lr_y_pred)
print("LR Mean squared error: ", lr_mse)
print("LR R-squared: ", lr_r2)
print("LR Mean absolute error: ", lr_mae)
print("LR Mean absolute error: ", lr_maep)

pip install xgboost

import xgboost as xgb

# XGBoost Regressor
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
xgb_y_pred = xgb_model.predict(X_test)

# Metrics for XGBoost
xgb_mse = mean_squared_error(y_test, xgb_y_pred)
xgb_r2 = r2_score(y_test, xgb_y_pred)
xgb_mae = mean_absolute_error(y_test, xgb_y_pred)
xgb_maep = mean_absolute_percentage_error(y_test, xgb_y_pred)

print("XGBoost Mean squared error: ", xgb_mse)
print("XGBoost R-squared: ", xgb_r2)
print("XGBoost Mean absolute error: ", xgb_mae)
print("XGBoost Mean absolute % error: ", xgb_maep)

import pandas as pd

# Define the metrics for each model
data = {
    'Model': ['Decision Tree', 'Random Forest', 'SVM', 'Linear Regression', 'XGBoost '],
    'Mean Squared Error': [det_mse, raf_mse, svm_mse, lr_mse, xgb_mse],
    'R-squared': [det_r2, raf_r2, svm_r2, lr_r2, xgb_r2],
    'Mean Absolute Error': [det_mae, raf_mae, svm_mae, lr_mae, xgb_mae],
    'Mean Absolute Percentage Error': [det_maep, raf_maep, svm_maep, lr_maep, xgb_maep]
}

# Create a DataFrame
results_df = pd.DataFrame(data)

# Print the DataFrame
print(results_df)

import matplotlib.pyplot as plt
import numpy as np

# Function to create the plot for actual vs. predicted values
def plot_predictions(y_true, y_pred, model_name, ax):
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'{model_name} Predictions')
    ax.set_aspect('equal')
    ax.grid(True)

# Create a figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # Adjust number of columns if needed

# Plot for Decision Tree
plot_predictions(y_test, det_y_pred, 'Decision Tree', axes[0, 0])

# Plot for Random Forest
plot_predictions(y_test, raf_y_pred, 'Random Forest', axes[0, 1])

# Plot for SVM
plot_predictions(y_test, svm_y_pred, 'SVM', axes[0, 2])

# Plot for Linear Regression
plot_predictions(y_test, lr_y_pred, 'Linear Regression', axes[1, 0])

# Plot for XGBoost
plot_predictions(y_test, xgb_y_pred, 'XGBoost', axes[1, 1])

# Hide empty subplot (if any)
axes[1, 2].axis('off')

# Improve spacing and layout
plt.tight_layout()
plt.show()

#saving our proposed model for deployment
import pickle
pickle.dump(raf_model, open('gpa_predict_model.pkl','wb'))