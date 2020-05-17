#Importing important dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


dataset = pd.read_csv("data.csv")



dataset.head()

sns.scatterplot(x="Height",y="Weight",data=dataset)


X=dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Feature scaling

from sklearn.preprocessing import StandardScaler
Sc_X = StandardScaler()
X = Sc_X.fit_transform(X)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
lin_reg.score(X,y)
#y1=lin_reg.predict(X)
#
#from sklearn.metrics import confusion_matrix
#confusion_matrix(y,y1)