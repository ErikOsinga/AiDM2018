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
	if normalization:
		print ("Normalization on")
		user_ratings_mean = np.mean(X, axis = 1)
		X = X - user_ratings_mean.reshape(-1, 1)

	else:
		print ("Normalization off")

	return X

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

	# Define a train/test mask on the array X 
	train_mask = np.ones((X.shape[0],X.shape[1]),dtype='bool')
	test_mask = np.ones((X.shape[0],X.shape[1]),dtype='bool')

	for k in range(len(train_users)):
		user, movie = train_users[k], train_movies[k]
		train_mask[user,movie] = False
	
	for k in range(len(test_users)):
		user, movie = test_users[k], test_movies[k]
		test_mask[user,movie] = False

	X_train = ma.array(np.copy(X),mask=train_mask)
	X_test = ma.array(np.copy(X),mask=test_mask)

	# Return the (user,movie) combination splits and train/test masked arrays
	return train_users, train_movies, test_users, test_movies, X_train, X_test

def get_RMSE(X,predictions):
	'''Calculate RMSE, X has to be a masked array'''
	return np.sqrt(np.mean(np.power(X - predictions,2)))

def get_MAE(X,predictions):
	'''Calculate MAE, X has to be a masked array'''
	return np.mean(np.abs(X - predictions)) 

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

	nfolds = 5

	# initialize
	X = create_X(ratings)

	# saving all folds
	# shape (nfolds,num_iter)
	all_all_RMSE_train = []
	all_all_RMSE_test = []
	
	all_all_MAE_train = []
	all_all_MAE_test = []

	for fold in range(nfolds):
		print ('Fold: %i'%fold)
		# split matrix X by removing 1/5th of the ratings as test set
		(train_users,train_movies,test_users,test_movies,
		X_train, X_test) = split_X_nfold(X,nfolds,fold)
		# We might lose some users or movie ratings totally in this split
		# but this is inevitable

		# initialize U and M 
		U = np.random.randn(len(all_users),num_factors)
		M = np.random.randn(num_factors,len(all_items))

		# For saving purposes
		all_RMSE_test = []
		all_RMSE_train = []
		all_MAE_test = []
		all_MAE_train = []

		for iterate in range(num_iter):
			print ('Iteration: %i'%iterate)

			# Fix M, solve (update) U by minimizing RMSE on train set
			for i in np.unique(train_users): # loop over all users still in train
				# indices of the movies that current user i has rated
				movies_user_i = np.where(X_train[i].mask^1)[0]
				n_ui = len(movies_user_i) # number of ratings of user i
				
				# Submatrix containing columns of the movies that user_i has rated
				M_Ii = M[:,movies_user_i].reshape(num_factors,n_ui)
				# row vector where columns j in Ii of the i-th row of X
				X_Ii = X_train[i][movies_user_i].reshape(n_ui,1)
				
				A_i = np.dot(M_Ii,M_Ii.T) + regularization * n_ui * np.eye(num_factors)
				V_i = np.dot(M_Ii,X_Ii)

				U[i,:] = np.dot(np.linalg.inv(A_i),V_i).reshape(num_factors,)

			predictions = np.dot(U,M)
			RMSE_test_1 = get_RMSE(X_test,predictions)
			print ('Test RMSE after solving for U:', RMSE_test_1)
			RMSE_train = get_RMSE(X_train,predictions)

			# Fix U, solve (update) M by minimizing RMSE on train set
			for j in np.unique(train_movies): # loop over all movies still in train
				# indices of the users that have rated current movie j
				users_movie_j = np.where(X_train[:,j].mask^1)[0]
				n_mj = len(users_movie_j) # number of users that rated this movie

				# Submatrix containing rows (columns) of the users that have rated this moive
				U_Ij = U[users_movie_j].reshape(n_mj,num_factors).T
				# column vector where rows i in Ij of the j-th column of X
				X_Ij = X_train[:,j][users_movie_j]

				A_j = np.dot(U_Ij,U_Ij.T) + regularization*n_mj * np.eye(num_factors)
				V_j = np.dot(U_Ij,X_Ij)

				M[:,j] = np.dot(np.linalg.inv(A_j),V_j).reshape(num_factors,)
			
			predictions = np.dot(U,M)
			RMSE_test = get_RMSE(X_test,predictions)
			print ('Test MSE after solving for M:', RMSE_test)
			RMSE_train = get_RMSE(X_train,predictions)

			all_RMSE_test.append(RMSE_test)
			all_RMSE_train.append(RMSE_train)
			all_MAE_train.append(get_MAE(X_train,predictions))
			all_MAE_test.append(get_MAE(X_test,predictions))

			if np.abs(RMSE_test_1 - RMSE_test) < bps:
				print ('RMSE has reached stopping criterion, stopping here.')
				break

		# save the result for this fold
		all_all_RMSE_test.append(all_RMSE_test)
		all_all_RMSE_train.append(all_RMSE_train)
		all_all_MAE_test.append(all_MAE_test)
		all_all_MAE_train.append(all_MAE_train)

	return all_all_RMSE_train, all_all_RMSE_test, all_all_MAE_train, all_all_MAE_test

normalization = True

all_RMSE_train, all_RMSE_test, all_MAE_train, all_MAE_test = ALS()

print ('Saving the results..')
if normalization:
	np.save('./all_RMSE_ALS_train_normalization',all_RMSE_train)
	np.save('./all_RMSE_ALS_test_normalization',all_RMSE_test)
	np.save('./all_MAE_ALS_train_normalization',all_MAE_train)
	np.save('./all_MAE_ALS_test_normalization',all_MAE_test)

else:
	np.save('./all_RMSE_ALS_train',all_RMSE_train)
	np.save('./all_RMSE_ALS_test',all_RMSE_test)
	np.save('./all_MAE_ALS_train',all_MAE_train)
	np.save('./all_MAE_ALS_test',all_MAE_test)

