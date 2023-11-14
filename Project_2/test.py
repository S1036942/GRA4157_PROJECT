import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm


# Read data:
def read_data(filename):
    dataframe = pd.read_csv(filename, sep = ",")
    return dataframe
df = read_data("Placement_Data_Full_Class.csv")

# Drop rows with missing 'salary' values for this regression analysis
df = df.dropna(subset=['salary'])


# Selecting features and target variable
#features = df[['gender', 'ssc_p', 'hsc_p', 'degree_p', 'workex', 'etest_p', 'specialisation', 'mba_p']]
#features = df[['gender', 'degree_p', 'workex', 'etest_p', 'specialisation', 'mba_p']]
#features = df[['gender', 'workex', 'hsc_s', 'specialisation', 'mba_p']]
#features = df[['gender','workex','mba_p']]
#features = df[['gender','mba_p']]
features = df[['workex','mba_p']]
target = df['salary']

# Convert categorical variables into dummy variables
features_encoded = pd.get_dummies(features, drop_first=True)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

"""
print(X_train.shape)    #Output here is: (118,8)
print(X_test.shape)     #Output here is: (30,8)
print(y_train.shape)    #Output here is: (118,)
print(y_test.shape)     #Output here is: (30,)
print(y_train)
print(X_train)
"""


X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

#X_train_sm['gender_M'] = X_train_sm['gender_M'].astype(int)
X_train_sm['workex_Yes'] = X_train_sm['workex_Yes'].astype(int)
#X_train_sm['specialisation_Mkt&HR'] = X_train_sm['specialisation_Mkt&HR'].astype(int)

#X_test_sm['gender_M'] = X_test_sm['gender_M'].astype(int)
X_test_sm['workex_Yes'] = X_test_sm['workex_Yes'].astype(int)
#X_test_sm['specialisation_Mkt&HR'] = X_test_sm['specialisation_Mkt&HR'].astype(int)

model = sm.OLS(y_train, X_train_sm).fit()
#model = sm.OLS(y_train, X_train).fit()
print(model.summary())

#Prediciton:
y_pred = model.predict(X_test_sm)
#y_pred = model.predict(X_test)


#Evaluating model:
#MSE:
mse = np.mean((y_test - y_pred) ** 2)
print(f"MSE: {mse:.3f}")
# R-squared:
r2 = model.rsquared
print(f"R-squared: {r2:.3f}")
#print(y_pred)





#new_data_point = {'gender':'M', 'ssc_p':'91.00', 'hsc_p':'60.00', 'degree_p':'70.00', 'workex':'Yes', 'etest_p':'66.00', 'specialisation':'Mkt&HR', 'mba_p':'88.00'}

### Do some prediction of the one data point here;