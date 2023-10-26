import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

def read_data(filename):
    dataframe = pd.read_csv(filename, sep = ",")
    return dataframe
df = read_data("Placement_Data_Full_Class.csv")


average_salary_by_gender = df.groupby(['gender', "degree_t"])["salary"].mean()
average_salary_by_gender_and_specialization = average_salary_by_gender.map("{:.2f}".format)
#print(average_salary_by_gender_and_specialization)

import sklearn
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
print(logreg)



import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

logreg = LogisticRegression()

# Drop rows with missing 'salary' values for this regression analysis
df_cleaned = df.dropna(subset=['salary'])

# Selecting features and target variable
features = df_cleaned[['gender', 'ssc_p', 'hsc_p', 'degree_p', 'workex', 'etest_p', 'specialisation', 'mba_p']]
target = df_cleaned['salary']

# Convert categorical variables into dummy variables
features_encoded = pd.get_dummies(features, drop_first=True)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(mse, r2)


from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
