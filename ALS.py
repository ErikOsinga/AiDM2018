import numpy as np
import numpy.ma as ma
import pandas as pd

# Array of shape (1000209,4) containing
# UserID::MovieID::Rating::Timestamp
ratings = np.loadtxt('./ml-1m/ratings.dat',delimiter="::")
all_users = np.unique(ratings[:,0])
all_items = np.unique(ratings[:,1])	 
np.random.seed(23)

def create_X(ratings):
	'''
	Create a masked array X that contains in position (i,j) 
	the rating that a user i would give a movie j
	'''
	ratings_df = pd.DataFrame(ratings, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
	X_df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating')
	X = X_df.as_matrix() # Shape = (6040, 3706)
	# only 5% of the matrix contains ratings
	X = ma.array(X,mask=np.isnan(X))

	# Can use different normalization methods (see paper)
	# Here we choose to subtract the mean rating of every user
	# We are not treating unrated movies as 0, because using a mask.
	print ("Normalization tbd")
	# user_ratings_mean = np.mean(X, axis = 1)
	# X = X - user_ratings_mean.reshape(-1, 1)

	return X

def get_RMSE(X,predictions):
	'''Calculate RMSE, X has to be a masked array'''
	return np.sqrt(np.mean(np.power(X - predictions,2)))

def ALS():
	'''
	Alternating least squares method to optimize U and M

	Step 1 Initialize matrix M by assigning the average rating for that movie as
		the first row, and small random numbers for the remaining entries.
	Step 2 Fix M, Solve U by minimizing the objective function (the sum of
		squared errors);
	Step 3 Fix U, solve M by minimizing the objective function similarly;
	Step 4 Repeat Steps 2 and 3 until a stopping criterion is satisfied.

	'''
	# Proposed parameters
	num_factors = 10 
	num_iter = 75
	regularization = 0.05
	learn_rate = 0.005
	bps = 1e-4

	# initialize
	X = create_X(ratings)
	U = np.random.randn(len(all_users),num_factors)
	M = np.random.randn(num_factors,len(all_items))

	# For saving purposes
	all_RMSE = []

	for iterate in range(num_iter):
		print ('Iteration: %i'%iterate)

		# Fix M, solve (update) U by minimizing RMSE
		for i in range(X.shape[0]):
			# indices of the movies that current user i has rated
			movies_user_i = np.where(X[i].mask^1)[0]
			n_ui = len(movies_user_i) # number of ratings of user i
			
			# Submatrix containing columns of the movies that user_i has rated
			M_Ii = M[:,movies_user_i].reshape(num_factors,n_ui)
			# row vector where columns j in Ii of the i-th row of X
			X_Ii = X[i][movies_user_i].reshape(n_ui,1)
			
			A_i = np.dot(M_Ii,M_Ii.T) + regularization * n_ui * np.eye(num_factors)
			V_i = np.dot(M_Ii,X_Ii)

			U[i,:] = np.dot(np.linalg.inv(A_i),V_i).reshape(num_factors,)

		RMSE = get_RMSE(X,np.dot(U,M))
		print ('MSE after solving for U:', RMSE)

		# Fix U, solve (update) M by minimizing RMSE
		for j in range(X.shape[1]):
			# indices of the users that have rated current movie j
			users_movie_j = np.where(X[:,j].mask^1)[0]
			n_mj = len(users_movie_j) # number of users that rated this movie

			# Submatrix containing rows (columns) of the users that have rated this moive
			U_Ij = U[users_movie_j].reshape(n_mj,num_factors).T
			# column vector where rows i in Ij of the j-th column of X
			X_Ij = X[:,j][users_movie_j]

			A_j = np.dot(U_Ij,U_Ij.T) + regularization*n_mj * np.eye(num_factors)
			V_j = np.dot(U_Ij,X_Ij)

			M[:,j] = np.dot(np.linalg.inv(A_j),V_j).reshape(num_factors,)
		
		RMSE = get_RMSE(X,np.dot(U,M))
		print ('MSE after solving for M:', RMSE)

		all_RMSE.append(RMSE)

		# TODO: update this rule to probe subset and a different rule
		try:
			if all_RMSE[-1] > all_RMSE[-2]:
				print ('RMSE has increased, stopping here.')
				break
		except IndexError:
			print ('First iteration done')

	return all_RMSE

all_RMSE = ALS()