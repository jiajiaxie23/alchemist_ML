from logistic_regression import Log_reg
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score



#from sklearn.linear_model import LogisticRegression

df = pd.read_excel('default_of_credit_card_clients.xls', header = 1)
df.dropna
X =  df.values[:, 1:23]
y = df.values[:, -1]

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

model = Log_reg(
		numerical_tol = 1e-5,
		alpha=1.0, 
		verbose=0,
		optim_method = 'CG'  ,
		random_init = False,
		bias_term = True
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












# model2 = LogisticRegression(random_state = 0).fit(X_train, y_train)

# y_hat = model2.predict(X_test)
# accur_test = accuracy_score(y_test, y_hat)


# print('Test Accuracy is {}'.format(accur_test))











