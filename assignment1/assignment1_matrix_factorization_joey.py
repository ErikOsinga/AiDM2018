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
	X_df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(np.nan)
	X = X_df.as_matrix() # Shape = (6040, 3706)

	# Can use different normalization methods (see paper)
	# Here we choose to subtract the mean rating of every user
	# Maybe better: Dont treat unrated movies as 0 
	X = np.ma.array(X, mask=np.isnan(X))
	user_ratings_mean = np.mean(X, axis = 1)
	X = X - user_ratings_mean.reshape(-1, 1)
	X = np.nan_to_num(X.data)
	

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
		print(X_train)
		print(X_test.shape)
		# Initializing from the standard normal dist
		U = np.random.rand(len(train_users),num_factors)
		M = np.random.rand(num_factors,len(train_items))
		U_t = U
		M_t = M

		prev_RMSE = 10e8
		for iterate in range(num_iter):
			# later: remove double for loop
			xhat_ij = np.dot(U,M)
			print(xhat_ij)
			e_ij = X_train - xhat_ij
#			print(e_ij)
#			for i in range(X_train.shape[0]):
#				print((i/X_train.shape[0])*100)
#				for j in range(X_train.shape[1]):
#					# prediction 
#					U[i,:] = U[i,:] + learn_rate * ( 2*e_ij[i,j] * M[:,j] - regularization * U[i,:] )
#					M[:,j] = M[:,j] + learn_rate * ( 2*e_ij[i,j] * U[i,:] - regularization * M[:,j] )
#			
#			print(U)
#			print(M)				
			er = np.empty(num_factors)	
			for i in range(X_train.shape[0]):
				for k in range(num_factors):
					er[k] = np.dot(e_ij[i,:], M[k,:])
					U_t[i,k] = U[i,k] + learn_rate * (2*er[k] - regularization*U[i,k] )
			for j in range(X_train.shape[1]):
				for k in range(num_factors):
					er[k]= np.dot(e_ij[:,j], U[:,k])
					M_t[k,j] = M[k,j] + learn_rate * (2*er[k] - regularization*M[k,j])
			U = U_t
			M = M_t
			print("Updated U and M")

			
			#change train to test
#			for i in range(X_train.shape[0]):
#				for j in range(X_train.shape[1]):
#					SE.append((X_test[i,j] - np.dot(U[i,:],M[:,j]))**2)
			RMSE = np.mean(np.abs(X_train-np.dot(U,M)))
			
			
			print ('Iteration: %i, RMSE = %f'%(iterate,RMSE))
			#RMSE = np.sqrt(np.mean(SE))
			if RMSE > prev_RMSE:
				print ('RMSE did not decrease, from %f to %f'%(prev_RMSE,RMSE))
				break
			prev_RMSE = RMSE


five_fold_CV_forloops()

