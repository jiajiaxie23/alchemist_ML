import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

class Support_Vector_Machine:

	def __init__(
		self,
		xi = 2, #relaxation parameters
		verbose =0
		):
		self.xi = xi
		self.verbose= verbose

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


		P,q,G,h,A,b = self._constr_mat(X,y)

		if self.verbose ==0:
			cvxopt_solvers.options['show_progress'] = False
		else:
			cvxopt_solvers.options['show_progress'] = True
		cvxopt_solvers.options['abstol'] = 1e-10
		cvxopt_solvers.options['reltol'] = 1e-10
		cvxopt_solvers.options['feastol'] = 1e-10

		#Run solver
		sol = cvxopt_solvers.qp(P, q, G, h, A, b)
		self._lambda= np.array(sol['x'])

		self._w = np.dot(X.T, self._lambda * y)
		self._b = np.sum(self._lambda * y)


		return self

	def predict(self, X):
		vals = np.dot(X, self._w) + self._b
		y_hat =np.where(vals >= 0., 1, -1)
		return y_hat

	def _constr_mat(self, X, y):

		N,d = X.shape
		hat_P = X.T * y
		y = y.reshape(-1,1) * 1.
		#print(hat_P.shape)
		P = cvxopt_matrix(np.dot(hat_P.T, hat_P))
		q=  cvxopt_matrix(- np.ones((N, 1)))
		h = cvxopt_matrix(np.zeros((N,1)))
		G = cvxopt_matrix(- np.eye(N))
		A = cvxopt_matrix(y.reshape(1, -1))
		b= cvxopt_matrix(np.zeros(1))





		# print(P.shape)
		# print(q.shape)
		# print(h.shape)
		# print(G.shape)
		#print(A.shape)
		# print(b.shape)

		return P, q, G, h, A, b

	# def _quadprog_solve_qp(self, P, q, G, h, A=None, b=None):
	# 	#qp_G = .5 * (P + P.T)   # make sure P is symmetric
	# 	qp_G = 0.5* P
	# 	qp_a = -q
	# 	if A is not None:
	# 		qp_C = -np.vstack([A, G]).T
	# 		qp_b = -np.hstack([b, h])
	# 		meq = A.shape[0]
	# 	else:  # no equality constraint
	# 		qp_C = -G.T
	# 		qp_b = -h
	# 		meq = 0
	# 	return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


	# def _cvxopt_solve_qp(self, P, q, G, h, A=None, b=None):
	# 	P = .5 * (P + P.T)  # make sure P is symmetric
	# 	args = [cvxopt.matrix(P), cvxopt.matrix(q)]
	# 	args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
	# 	if A is not None:
	# 		args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
	# 	sol = cvxopt.solvers.qp(*args)
	# 	if 'optimal' not in sol['status']:
	# 		return None
	# 	return np.array(sol['x']).reshape((P.shape[1],))





