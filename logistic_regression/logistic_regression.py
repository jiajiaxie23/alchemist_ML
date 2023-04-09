import numpy as np
from scipy.optimize import minimize
from scipy.special import expit as _sigmoid

def _gradient(beta, X, y):
	an = _sigmoid(np.dot(X, beta))
	#print(an.shape)
	grad = np.dot(an.T - y.T, X)
	#print(grad.shape)
	return grad.T

def _func(beta, X, y, esp = 0.0001):
	an = _sigmoid(np.dot(X, beta))
	ell = np.log(an + esp)
	one_minus_ell = np.log(1- an  + esp)
	return -(np.dot(y , ell) + np.dot(1-y, one_minus_ell))


class Log_reg:

	def __init__(
		self,
		numerical_tol = 1e-5,
		alpha=1.0,  #regularization weight
		verbose=0,
		optim_method = 'CG'  , #conjugate gradident by default
		random_init = False,
		bias_term = True ,# Add bias term or not
		lr = 0.005 #learning rate
		):
		self.alpha = alpha
		self.verbose = verbose
		self.optim_method = optim_method
		self.random_init = random_init
		self.bias_term = bias_term
		self.numerical_tol = numerical_tol
		self.lr = lr

	def fit(self, X, y):

		"""Fit a mixed Negative Binomial Models.
		Parameters
		----------
		X : {array-like, sparse matrix} of shape (N, d)
		Training data.
		y : array-like of shape (N,)
		Target values.
		sample_weight : array-like of shape (N,), default=None
		Sample weights.
		Returns
		-------
		self : object
		Fitted model.
		"""


		X = self._init_params(X)

		prev_obj = np.inf
		converged = False
		while converged == False:

		#for ii in range(3):
			up_beta = self.beta - self.lr * _gradient(self.beta, X,y)
			obj = _func(up_beta, X,y)[0]
			print("The current objective is {}".format(obj))

			if obj - prev_obj >= 0:
				converged = True
				print("Gradient Descent Converged")

			else:


				prev_obj = obj
				self.beta = up_beta


		# res = minimize(_func,  
		# 			x0 = self.beta,
		# 			jac = _gradient, 
		# 			args = (X, y),
		# 			method=self.optim_method,
		# 			tol = self.numerical_tol)



		return self

	def predict(self, X):
		N,d  = X.shape
		if self.bias_term == True:
			z = np.ones((N,1))
			X = np.append(X, z, axis=1)
		an = _sigmoid(np.dot(X, self.beta))
		y_hat =np.where(an >= 0.5, 1, 0)
		return y_hat

	def _init_params(self,X):

		N,d = X.shape

		if self.bias_term == True:
			d = d +1
			z = np.ones((N,1))
			X = np.append(X, z, axis=1)

		if self.random_init == True:
			self.beta =  np.random.randn(d, 1)
		else:
			self.beta = np.ones((d,1))

		return X





























