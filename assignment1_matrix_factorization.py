import numpy as np
import pandas as pd

# Array of shape (1000209,4) containing
# UserID::MovieID::Rating::Timestamp
ratings = np.loadtxt('./ml-1m/ratings.dat',delimiter="::")
all_users = np.unique(ratings[:,0])
all_items = np.unique(ratings[:,1])	 

def create_X(ratings):
	'''
	Create a matrix X that contains in position (i,j) 
	the rating that a user i would give a movie j
	'''
	ratings_df = pd.DataFrame(ratings, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
	X_df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
	X = X_df.as_matrix() # Shape = (6040, 3706)

	# Can use different normalization methods (see paper)
	# Here we choose to subtract the mean rating of every user
	# Maybe better: Dont treat unrated movies as 0 
	user_ratings_mean = np.mean(X, axis = 1)
	X = X - user_ratings_mean.reshape(-1, 1)

	return X


def five_fold_CV_forloops():

	# Proposed parameters
	num_factors = 10 
	num_iter = 75
	regularization = 0.05
	learn_rate = 0.005

	nfolds = 2
	for fold in range(nfolds):
		np.random.seed(23)
		print ('Fold number: %i'%fold)
		seqs=[x%nfolds for x in range(len(ratings))]
		np.random.shuffle(seqs)
		
		train_sel = np.array([x!=fold for x in seqs])
		test_sel = np.array([x==fold for x in seqs])
		train = ratings[train_sel]
		test = ratings[test_sel]
		
		train_items = np.unique(train[:,1]) 
		train_users = np.unique(train[:,0])

		X_train = create_X(train) 
		X_test = create_X(test)

		# Initializing from the standard normal dist
		U = np.random.randn(len(train_users),num_factors)
		M = np.random.randn(num_factors,len(train_items))

		prev_SE = 10e8
		for iterate in range(num_iter):
			# later: remove double for loop
			for i in range(X_train.shape[0]):
				for j in range(X_train.shape[1]):
					# prediction 
					xhat_ij = np.dot(U[i,:],M[:,j])
					e_ij = X_train[i,j] - xhat_ij
					for k in range(num_factors):
						U[i,k] = U[i,k] + learn_rate * ( 2*e_ij * M[k,j] - regularization * U[i,k] )
						M[k,j] = M[k,j] + learn_rate * ( 2*e_ij * U[i,k] - regularization * M[k,j] )

			SE = 0
			for i in range(X_test.shape[0]):
				for j in range(X_test.shape[1]):
					SE += np.power(X_test[i,j] - np.dot(U[i,:],M[:,j]),2)
			print ('Iteration: %i, SE = %f'%(iterate,SE))
			if SE > prev_SE:
				print ('SE did not decrease, from %f to %f'%(prev_SE,SE))
				break
			prev_SE = SE


five_fold_CV_forloops()

