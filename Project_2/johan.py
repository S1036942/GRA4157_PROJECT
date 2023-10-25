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


