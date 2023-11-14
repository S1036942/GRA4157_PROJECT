import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # graphs potting 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, classification_report
from statsmodels.api import OLS

import numpy as np # linear algebra
import pandas as pd

def read_data(filename):
    dataframe = pd.read_csv(filename, sep = ",")
    return dataframe
df = read_data("Placement_Data_Full_Class.csv")

X=df['degree_p'].values.reshape(-1,1)
y=df['mba_p'].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()

reg.fit(X_train,y_train)

#predicting the test set results
y_pred=reg.predict(X_test)

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,reg.predict(X_train),color='blue')
plt.title('prediction on the training set')
plt.xlabel('Degree_p')
plt.ylabel('mba_p')
plt.show()
