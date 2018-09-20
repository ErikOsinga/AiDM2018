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

def split_X_nfold_forloops(X,nfolds,fold):
	'''
	Split X in a train and test set
	To split in 5 folds we mask 1/5 of what is now unmasked in X
	'''
	train_mask = np.copy(X.mask)
	test_mask = np.copy(X.mask)

	# all (user,movie) pairs that exist in the ratings.dat file
	# are combinations of (i[k],j[k]) for k in (0,len(ratings))
	i, j = np.where(X.mask == False)

	seqs=[x%nfolds for x in range(len(ratings))]
	np.random.shuffle(seqs)

	# array of len(ratings) containing 4/5*len(ratings) True's
	train_sel = np.array([x!=fold for x in seqs])
	train_users = i[train_sel]
	train_movies = j[train_sel]

	# test sel is then the other 1/5 of the ratings.
	test_sel = np.array([x==fold for x in seqs])
	test_users = i[test_sel]
	test_movies = j[test_sel]

	# remove the test_users from the training set by masking it 
	for k in range(len(test_users)):
		user, movie = test_users[k], test_movies[k]
		train_mask[user,movie] = True

	# remove the train users from the test set by masking them
	for k in range(len(train_users)):
		user, movie = train_users[k], train_movies[k]
		test_mask[user,movie] = True

	X_train = ma.array(X,mask=train_mask)
	X_test = ma.array(X,mask=test_mask)

	return X_train, X_test

def split_X_nfold(X,nfolds,fold):
	'''
	Split X in a train and test set
	To split in 5 folds we simply use the indices of the 
	(user,movie) combinations that exist in X
	split these in 4/5 and 1/5 and return these indices
	'''

	# all (user,movie) pairs that exist in the ratings.dat file
	# are combinations of (i[k],j[k]) for k in (0,len(ratings))
	i, j = np.where(X.mask == False)

	seqs=[x%nfolds for x in range(len(ratings))]
	np.random.shuffle(seqs)

	# array of len(ratings) containing 4/5*len(ratings) True's
	train_sel = np.array([x!=fold for x in seqs])
	train_users = i[train_sel]
	train_movies = j[train_sel]

	# test sel is then the other 1/5 of the ratings.
	test_sel = np.array([x==fold for x in seqs])
	test_users = i[test_sel]
	test_movies = j[test_sel]

	'''
	# e.g., for looping over  all training (user,movie) combinations:
	for k in range(len(train_users)):
		user, movie = train_users[k], train_movies[k]
	'''

	# Return the (user,movie) combination splits
	return train_users, train_movies, test_users, test_movies

def five_fold_CV_forloops():

	# Proposed parameters
	num_factors = 10 
	num_iter = 75
	regularization = 0.05
	learn_rate = 0.005

	X = create_X(ratings)

	nfolds = 5
	for fold in range(nfolds):
		print ('Fold number: %i'%fold)
	
		X_train, X_test = split_X_nfold_forloops(X,nfolds,fold)

		# Initializing from the standard normal dist
		U = np.random.randn(len(all_users),num_factors)
		M = np.random.randn(num_factors,len(all_items))

		prev_SE = 10e8
		for iterate in range(num_iter):
			# later: remove double for loop
			for i in range(X_train.shape[0]):
				for j in range(X_train.shape[1]):
					if not ma.is_masked(X_train[i,j]): # only predict if the value is not masked
						# prediction 
						xhat_ij = np.dot(U[i,:],M[:,j])
						e_ij = X_train[i,j] - xhat_ij
						for k in range(num_factors):
							U[i,k] = U[i,k] + learn_rate * ( 2*e_ij * M[k,j] - regularization * U[i,k] )
							M[k,j] = M[k,j] + learn_rate * ( 2*e_ij * U[i,k] - regularization * M[k,j] )

			SE = 0
			for i in range(X_test.shape[0]):
				for j in range(X_test.shape[1]):
					if not ma.is_masked(X_test[i,j]): # only predict if the value is not masked
						SE += np.power(X_test[i,j] - np.dot(U[i,:],M[:,j]),2)
			
			print ('Iteration: %i, SE = %f'%(iterate,SE))
			if SE > prev_SE:
				print ('SE did not decrease, from %f to %f'%(prev_SE,SE))
				print ('Thus stopping fold number %i \n'%fold)
				break
			prev_SE = SE

# five_fold_CV_forloops()

def five_fold_CV():

	# Proposed parameters
	num_factors = 10 
	num_iter = 75
	regularization = 0.05
	learn_rate = 0.005

	X = create_X(ratings)

	nfolds = 5
	for fold in range(nfolds):
		print ('Fold number: %i'%fold)
	
		train_users,train_movies,test_users,test_movies = split_X_nfold(X,nfolds,fold)

		# Initializing from the standard normal dist
		U = np.random.randn(len(all_users),num_factors)
		M = np.random.randn(num_factors,len(all_items))

		prev_SE = 10e8
		for iterate in range(num_iter):
			
			# loop through the training (user,movie) combinations
			for l in range(len(train_users)):
				i, j = train_users[l], train_movies[l]
				xhat_ij = np.dot(U[i,:],M[:,j])
				e_ij = X[i,j] - xhat_ij

				for k in range(num_factors):
					U[i,k] = U[i,k] + learn_rate * ( 2*e_ij * M[k,j] - regularization * U[i,k] )
					M[k,j] = M[k,j] + learn_rate * ( 2*e_ij * U[i,k] - regularization * M[k,j] )

			SE = 0
			# loop through the test (user,movie) combinations
			for l in range(len(test_users)):
				i, j = test_users[l], test_movies[l]
				SE += np.power(X[i,j] - np.dot(U[i,:],M[:,j]),2)
			
			# Check if SE decreased
			print ('Iteration: %i, SE = %f'%(iterate,SE))
			if SE > prev_SE:
				print ('SE did not decrease, from %f to %f'%(prev_SE,SE))
				print ('Thus stopping fold number %i \n'%fold)
				break
			prev_SE = SE

	return U, M

U, M = five_fold_CV()
