from svm import Support_Vector_Machine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import numpy as np


#from sklearn.linear_model import LogisticRegression

df = pd.read_excel('../default_of_credit_card_clients.xls', header = 1)
df.dropna
X =  df.values[:, 1:23]
y = df.values[:, -1]
y = np.where(y ==0, -1, y)

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

model = Support_Vector_Machine(
		xi = 2,
		verbose = 1
	)





X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.33, random_state=42)




model.fit(X_train, y_train)



y_hat =model.predict(X_test)
accur_test = accuracy_score(y_test, y_hat)


y_hat_train =model.predict(X_train)
accur_train = accuracy_score(y_train, y_hat_train)

print('Train Accuracy is {}'.format(accur_train))


print('Test Accuracy is {}'.format(accur_test))