import numpy as np
import numpy.ma as ma
import pandas as pd

# Array of shape (1000209,4) containing
# UserID::MovieID::Rating::Timestamp
ratings = np.loadtxt('./ml-1m/ratings.dat',delimiter="::")
all_users = np.unique(ratings[:,0])
all_items = np.unique(ratings[:,1])	 
np.random.seed(23)

def create_X(ratings,normalization):
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

def five_fold_CV():

	# Proposed parameters
	num_factors = 10 
	num_iter = 75
	regularization = 0.05
	learn_rate = 0.005

	X = create_X(ratings)

	# For saving results
	all_U = []
	all_M = []
	all_RMSE_train = []
	all_MAE_train = []
	all_RMSE_test = []
	all_MAE_test = []

	nfolds = 5
	for fold in range(nfolds):
		print ('Fold number: %i'%fold)
	
		(train_users,train_movies,test_users,test_movies,
		X_train, X_test) = split_X_nfold(X,nfolds,fold)

		# Initializing from the standard normal dist
		U = np.random.randn(len(all_users),num_factors)
		M = np.random.randn(num_factors,len(all_items))

		prev_RMSE = 10e8
		for iterate in range(num_iter):
			
			# loop through the training (user,movie) combinations
			for l in range(len(train_users)): # this loop cannot be avoided
				i, j = train_users[l], train_movies[l]
				xhat_ij = np.dot(U[i,:],M[:,j]) # have to keep recalculating in loop
				e_ij = X[i,j] - xhat_ij # because we keep updating U and M

				# update all 10 factors beloning to (i,j) of this training example
				U[i,:] = U[i,:] + learn_rate * ( 2*e_ij * M[:,j] - regularization * U[i,:] )
				M[:,j] = M[:,j] + learn_rate * ( 2*e_ij * U[i,:] - regularization * M[:,j] )
				
			# all predictions after the updates
			predictions = np.dot(U,M)

			# calculate RMSE on test set for the stopping condition
			RMSE_test = np.sqrt(np.mean(np.power(X_test - predictions,2)))
			
			# Check if RMSE decreased
			print ('Iteration: %i, test set RMSE = %f'%(iterate,RMSE_test))
			if RMSE_test > prev_RMSE or iterate == (num_iter - 1):
				if iterate == (num_iter - 1):
					print ('Last iteration done, stopping..')
				else:
					print ('RMSE did not decrease, from %f to %f'%(prev_RMSE,RMSE_test))
					print ('Thus stopping fold number %i \n'%fold)
				
				# save results of this fold
				all_RMSE_test.append(RMSE_test)
				all_RMSE_train.append( np.sqrt(np.mean(np.power(X_train - predictions,2))) )
				
				all_MAE_test.append( np.mean(np.abs(X_test - predictions)) )
				all_MAE_train.append( np.mean(np.abs(X_train - predictions)) )
				
				all_U.append(U)
				all_M.append(M)

				break
				
			prev_RMSE = RMSE_test

	return all_U, all_M, all_RMSE_train, all_RMSE_test, all_MAE_train, all_MAE_test

def show_learning_curve():
	''' 
	Only runs 1 fold of the above algorithm to save the
	MAE and RMSE at every iteration to show the learning curve.
	'''

	# Proposed parameters
	num_factors = 10 
	num_iter = 75
	regularization = 0.05
	learn_rate = 0.005

	X = create_X(ratings)

	# For saving results per iteration
	RMSE_train_it = []
	MAE_train_it = []
	RMSE_test_it = []
	MAE_test_it = []

	nfolds = 5
	for fold in range(1):
		print ('Only doing 1 fold, calculating error per iteration...')
		print ('Fold number: %i'%fold)
	
		(train_users,train_movies,test_users,test_movies,
		X_train, X_test) = split_X_nfold(X,nfolds,fold)

		# Initializing from the standard normal dist
		U = np.random.randn(len(all_users),num_factors)
		M = np.random.randn(num_factors,len(all_items))

		prev_RMSE = 10e8
		for iterate in range(num_iter):
			
			# loop through the training (user,movie) combinations
			for l in range(len(train_users)): # this loop cannot be avoided
				i, j = train_users[l], train_movies[l]
				xhat_ij = np.dot(U[i,:],M[:,j]) # have to keep recalculating in loop
				e_ij = X[i,j] - xhat_ij # because we keep updating U and M

				# update all 10 factors beloning to (i,j) of this training example
				U[i,:] = U[i,:] + learn_rate * ( 2*e_ij * M[:,j] - regularization * U[i,:] )
				M[:,j] = M[:,j] + learn_rate * ( 2*e_ij * U[i,:] - regularization * M[:,j] )
				
			# all predictions after the updates
			predictions = np.dot(U,M)

			# calculate RMSE on test set for the stopping condition
			RMSE_test = np.sqrt(np.mean(np.power(X_test - predictions,2)))
			
			# save results of this fold
			RMSE_test_it.append(RMSE_test)
			RMSE_train_it.append( np.sqrt(np.mean(np.power(X_train - predictions,2))) )
			
			MAE_test_it.append( np.mean(np.abs(X_test - predictions)) )
			MAE_train_it.append( np.mean(np.abs(X_train - predictions)) )
		
			# Check if RMSE decreased
			print ('Iteration: %i, test set RMSE = %f'%(iterate,RMSE_test))
			if RMSE_test > prev_RMSE:
				print ('RMSE did not decrease, from %f to %f'%(prev_RMSE,RMSE_test))
				print ('Thus stopping fold number %i \n'%fold)				
				break
			prev_RMSE = RMSE_test

	return RMSE_train_it, RMSE_test_it, MAE_train_it, MAE_test_it

normalization = False
all_U, all_M, all_RMSE_train, all_RMSE_test, all_MAE_train, all_MAE_test = five_fold_CV()
if normalization:
	np.save('./all_U_normalization',all_U)
	np.save('./all_M_normalization',all_M)
	np.save('./all_RMSE_train_normalization',all_RMSE_train)
	np.save('./all_RMSE_test_normalization',all_RMSE_test)
	np.save('./all_MAE_train_normalization',all_MAE_train)
	np.save('./all_MAE_test_normalization',all_MAE_test)

else:
	np.save('./all_U',all_U)
	np.save('./all_M',all_M)
	np.save('./all_RMSE_train',all_RMSE_train)
	np.save('./all_RMSE_test',all_RMSE_test)
	np.save('./all_MAE_train',all_MAE_train)
	np.save('./all_MAE_test',all_MAE_test)



# # For plotting a learning curve
# RMSE_train_it, RMSE_test_it, MAE_train_it, MAE_test_it = show_learning_curve()
# np.save('./RMSE_test_it',RMSE_test_it)
# np.save('./RMSE_train_it',RMSE_train_it)
# np.save('./MAE_train_it',MAE_train_it)
# np.save('./MAE_test_it',MAE_test_it)