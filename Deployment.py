import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df = pd.read_csv(r'C:\Users\admin\Desktop\Loan_Predictions\train_data.csv')

print(df.columns)
# Info about the columns
# print(df.info())
# print(df.describe())

# As id column is not important so drop it
df.drop(columns=['Loan_ID'], axis=1, inplace=True)

features_with_na = [feature for feature in df.columns if df[feature].isnull().sum()>0]

# for feature in features_with_na:
#     df[feature].hist(bins=20)
#     plt.xlabel(feature)
#     plt.ylabel('Count')
#     plt.show()

# Missing values
# Replacing each column missing values with their mode values

# Filling the missing values

for feature in features_with_na:
    df[feature].fillna(df[feature].dropna().mode().values[0], inplace=True)

# Data visualization

# creating the list of numeric and categorical features seperately

# cat = df.select_dtypes('O').columns.to_list()
categorical_feature = [feature for feature in df.columns if df[feature].dtype == 'O']

# num = df.select_dtypes('number').columns.to_list()
numerical_feature = [feature for feature in df.columns if df[feature].dtype != 'O']

# Ploting for numerical features
# for feature in numerical_feature:
#     df[feature].hist(bins=20)
#     plt.xlabel(feature)
#     plt.ylabel('Count')
#     plt.show()


# total = float(len(df[categorical_feature[-1]]))
# plt.figure(figsize=(6, 8))
# sns.set(style='whitegrid')
# ax = sns.countplot(df[categorical_feature[-1]])
# for p in ax.patches:
#     height = p.get_height()
#     ax.text(p.get_x() + p.get_width()/2., height+3, '{:1.2f}'.format(height/total), ha='center')
# plt.show()

# Ploting for categorical feature

# for feature in categorical_feature[:-1]:
#     sns.countplot(x=df[feature], hue=df['Loan_Status'], palette='plasma')
#     plt.xlabel(feature, fontsize=20)
#     plt.show()

# Encoding data to numeric
# for feature in categorical_feature:
#     df[feature] = pd.get_dummies(df[feature], drop_first=True)
# pd.set_option('display.max_columns', 12)
# print(df.head(15))
# print(df.dtypes)

to_numeric = {'Male': 1, 'Female': 2, 'Yes': 1, 'No': 2, 'Graduate': 1, 'Not Graduate': 2,
              'Urban': 3, 'Semiurban': 2, 'Rural': 1,
              'Y': 1, 'N': 0,
              '3+': 3}
df = df.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)

# Converting the dependent column
df['Dependents_'] = pd.to_numeric(df['Dependents'])
df.drop('Dependents', axis=1, inplace=True)

# Correlaton matrix

corr = df.corr()

# plt.figure(figsize=(8, 10))
# g = sns.heatmap(corr, cmap='cubehelix_r', annot=True)
# plt.xticks(rotation=30)
# g.set_xticklabels(df.columns, rotation=30)
# plt.show()

# Correlation table for a more detailed analysis

# corr.style.background_gradient(cmap='coolwarm').set_precision(2)

# we find out that credit history is highly correlated to loan status as compared to other indp. variables

# ML model creation

# Models we will use:

# Decision Tree
# Random Forest
# XGBoost
# Logistic Regression

# Decision Tree
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# from sklearn.tree import DecisionTreeClassifier
# dt_classifier = DecisionTreeClassifier()
# dt_classifier.fit(X_train, y_train)
# y_pred = dt_classifier.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# print(accuracy_score(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

# score = cross_val_score(dt_classifier, X, y, scoring='accuracy', cv=5)
# print(score.mean())

# Random Forest
# from sklearn.ensemble import RandomForestClassifier
# rf_classifier = RandomForestClassifier(n_estimators=110)
#
# rf_classifier.fit(X_train, y_train)
# y_pred = rf_classifier.predict(X_test)
#
# print(accuracy_score(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

# score = cross_val_score(rf_classifier, X, y, scoring='accuracy', cv=5)
# print(score.max())

# parameter = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,150]}
#
# grid_score = GridSearchCV(rf_classifier, param_grid=parameter, scoring='accuracy', cv=5)
# grid_score.fit(X, y)
# print(grid_score.best_params_)
# print(grid_score.best_score_)

# Xgboost Classifier
import xgboost
# from xgboost import XGBClassifier
# xgb_classifier = XGBClassifier(use_label_encoder=False)
#
# xgb_classifier.fit(X_train, y_train)
# y_pred = xgb_classifier.predict(X_test)

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression()

lr_classifier.fit(X_train, y_train)
# y_pred = lr_classifier.predict(X_test)

# Create a pickle file for logistic regression
filename = r'C:\Users\admin\Desktop\lr_classifier_model.pkl'
pickle.dump(lr_classifier, open(filename, 'wb'))

#
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))

# SVM classifier
# from sklearn.svm import SVC
# svm_classifier = SVC()

# svm_classifier.fit(X_train, y_train)
# y_pred = svm_classifier.predict(X_test)

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))

# Now lets test for test data

# df_test = pd.read_csv(r'C:\Users\admin\Desktop\Loan_Predictions\test_data.csv')
#
# # As id column is not important so drop it
# df_test.drop(columns=['Loan_ID'], axis=1, inplace=True)
#
# features_with_na_test = [feature for feature in df_test.columns if df_test[feature].isnull().sum()>0]
#
# for feature in features_with_na_test:
#     df_test[feature].fillna(df_test[feature].dropna().mode().values[0], inplace=True)
#
# to_numerical = {'Male': 1, 'Female': 2, 'Yes': 1, 'No': 2, 'Graduate': 1, 'Not Graduate': 2,
#               'Urban': 3, 'Semiurban': 2, 'Rural': 1,
#               'Y': 1, 'N': 0,
#               '3+': 3}
# df_test = df_test.applymap(lambda lable: to_numerical.get(lable) if lable in to_numerical else lable)
#
# df_test['Dependents_'] = pd.to_numeric(df_test['Dependents'])
# df_test.drop('Dependents', axis=1, inplace=True)
#
# test_data = df_test
# y_pred = lr_classifier.predict(test_data)
# # print(y_pred)
