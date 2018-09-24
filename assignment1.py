import numpy as np
from sklearn import linear_model

# Array of shape (1000209,4) containing
# UserID::MovieID::Rating::Timestamp
ratings = np.loadtxt('./ml-1m/ratings.dat',delimiter="::")

all_users = np.unique(ratings[:,0])
all_items = np.unique(ratings[:,1])	 

# Array of shape 
# MovieID::Title::Genres
# movies = np.loadtxt('./ml-1m/movies.dat',delimiter="::",dtype='str')

def global_mean(ratings):
	'''Find mean over all ratings of all users '''
	prediction = np.mean(ratings[:,2])
	return prediction
	
def item_mean(ratings,item):
	'''
	Find mean of all ratings of 1 item
	Input: ratings array and item number
	'''
	item_ratings = ratings[ratings[:,1] == item]
	prediction = np.mean(item_ratings[:,2])
	return prediction
	
def user_mean(ratings,user):
	'''Find mean of all ratings of 1 user
	Input: ratings array and user number
	'''
	user_ratings = ratings[ratings[:,0] == user]
	prediction = np.mean(user_ratings[:,2])
	return prediction

def five_fold_CV(ratings):
	'''
	Test the models with 5 fold cross validation
	'''
	
	np.random.seed(17) # For reproducibility
	
	# split data into 5 train and test folds
	nfolds=5

	# allocate memory for results:
	# linear combination of the 3 averages
	err_train_LR = np.zeros(nfolds)
	err_train_LR_MAE = np.zeros(nfolds)
	err_test_LR = np.zeros(nfolds) 
	err_test_LR_MAE = np.zeros(nfolds) 
	# Rating = user mean
	err_test_UM = np.zeros(nfolds)
	err_test_UM_MAE = np.zeros(nfolds)
	# Rating = item mean
	err_test_IM = np.zeros(nfolds)
	err_test_IM_MAE = np.zeros(nfolds)
	# Rating = Global Mean
	err_test_GM = np.zeros(nfolds)
	err_test_GM_MAE = np.zeros(nfolds)

	# allocate memory for train error results
	# Rating = user mean
	err_train_UM = np.zeros(nfolds)
	err_train_UM_MAE = np.zeros(nfolds)
	# Rating = item mean
	err_train_IM = np.zeros(nfolds)
	err_train_IM_MAE = np.zeros(nfolds)
	# Rating = Global Mean
	err_train_GM = np.zeros(nfolds)
	err_train_GM_MAE = np.zeros(nfolds)
	
	alpha = np.zeros(nfolds)
	beta = np.zeros(nfolds)
	gamma = np.zeros(nfolds)
						
	seqs=[x%nfolds for x in range(len(ratings))]
	np.random.shuffle(seqs)

	def user_mean_error():
		err_test_UM[fold] = np.sqrt(np.mean(np.power(test[:,2] - test_prediction_UM[:,1],2)))
		err_test_UM_MAE[fold] = np.mean(np.abs(test[:,2] - test_prediction_UM[:,1]))
		err_train_UM[fold] = np.sqrt(np.mean(np.power(train[:,2] - train_prediction_UM[:,1],2)))
		err_train_UM_MAE[fold] = np.mean(np.abs(train[:,2] - train_prediction_UM[:,1]))

	def item_mean_error():
		err_test_IM[fold] = np.sqrt(np.mean(np.power(test[:,2] - test_prediction_IM[:,1],2)))
		err_test_IM_MAE[fold] = np.mean(np.abs(test[:,2] - test_prediction_IM[:,1]))
		err_train_IM[fold] = np.sqrt(np.mean(np.power(train[:,2] - train_prediction_IM[:,1],2)))
		err_train_IM_MAE[fold] = np.mean(np.abs(train[:,2] - train_prediction_IM[:,1]))

	def global_mean_error():
		err_test_GM[fold] = np.sqrt(np.mean( (test[:,2] - global_mean_train)**2 ))
		err_test_GM_MAE[fold] = np.mean(np.abs( test[:,2] - global_mean_train ))
		err_train_GM[fold] = np.sqrt(np.mean( (train[:,2] - global_mean_train)**2 ))
		err_train_GM_MAE[fold] = np.mean(np.abs( train[:,2] - global_mean_train ))
	
	def train_linear_combination(train):
		# Find all user means and all item means
		y = train[:,2]
		# construct the x1,x2 matrix
		x1 = train_prediction_UM[:,1]
		x2 = train_prediction_IM[:,1]

		X = np.asarray([x1,x2]).T
		reg = linear_model.LinearRegression()
		reg.fit(X,y)
		alpha, beta = reg.coef_
		gamma = reg.intercept_
		
		return alpha,beta,gamma

	#for each fold:
	for fold in range(nfolds):
		print ('Fold number: %i'%fold)
		
		train_sel=np.array([x!=fold for x in seqs])
		test_sel=np.array([x==fold for x in seqs])
		train=ratings[train_sel]
		test=ratings[test_sel]
		
		train_items = np.unique(train[:,1]) 
		train_users = np.unique(train[:,0]) 

		# matrix of shape (len(test),2) containing all test/train users/items and their prediction ratings
		test_prediction_UM = np.array([test[:,0],np.empty(len(test[:,0]))]).T
		test_prediction_IM = np.array([test[:,1],np.empty(len(test[:,1]))]).T
		train_prediction_UM = np.array([train[:,0],np.empty(len(train[:,0]))]).T
		train_prediction_IM = np.array([train[:,1],np.empty(len(train[:,1]))]).T
		
		global_mean_train = global_mean(train)

		for user in all_users:
			if user in train_users: 
				user_mean_user = user_mean(train,user) # revalculate user and item mean on the train set
				# create a matrix with test predictions for quicker calculation of error
				test_prediction_UM[test_prediction_UM[:,0] == user, 1] = user_mean_user
				train_prediction_UM[train_prediction_UM[:,0] == user, 1] = user_mean_user
			else: # user is not in train set, use global mean
				test_prediction_UM[test_prediction_UM[:,0] == user, 1] = global_mean_train
		for item in all_items:
			if item in train_items: 
				item_mean_item = item_mean(train,item)
				# create a matrix with test predictions for quicker calculation of error
				test_prediction_IM[test_prediction_IM[:,0] == item, 1] = item_mean_item
				train_prediction_IM[train_prediction_IM[:,0] == item, 1] = item_mean_item
			else: # user is not in train set, use global mean
				test_prediction_IM[test_prediction_IM[:,0] == item, 1] = global_mean_train
		
		# Very naive approaches
		user_mean_error()
		item_mean_error()
		global_mean_error()

		# Linear Regression
		alpha[fold], beta[fold], gamma[fold] = train_linear_combination(train)
		LR_predict_train = alpha[fold] * train_prediction_UM[:,1] + beta[fold] * train_prediction_IM[:,1] + gamma[fold]
		LR_predict_test = alpha[fold] * test_prediction_UM[:,1] + beta[fold] * test_prediction_IM[:,1] + gamma[fold]

		err_train_LR[fold] = np.sqrt(np.mean((train[:,2] - LR_predict_train)**2))
		err_test_LR[fold] = np.sqrt(np.mean((test[:,2] - LR_predict_test )**2))

		err_train_LR_MAE[fold] = np.mean(np.abs((train[:,2] - LR_predict_train)**2))
		err_test_LR_MAE[fold] = np.mean(np.abs((test[:,2] - LR_predict_test )**2))

	return (err_train_LR, err_train_LR_MAE, err_test_LR, err_test_LR_MAE, err_test_UM, err_test_UM_MAE,
			err_test_IM, err_test_IM_MAE, err_test_GM, err_test_GM_MAE,
			err_train_UM, err_train_UM_MAE, err_train_IM, err_train_IM_MAE,
			err_train_GM, err_train_GM_MAE, alpha, beta, gamma)

	
#alpha,beta,gamma = train_linear_combination(ratings)
# for quick testing purposes
# alpha = 0.7811833
# beta = 0.87454303
# gamma = -2.3489102231862939

(err_train_LR, err_train_LR_MAE, err_test_LR, err_test_LR_MAE, err_test_UM, err_test_UM_MAE,
err_test_IM, err_test_IM_MAE, err_test_GM, err_test_GM_MAE,
err_train_UM, err_train_UM_MAE, err_train_IM, err_train_IM_MAE,
err_train_GM, err_train_GM_MAE, alpha, beta, gamma) = five_fold_CV(ratings)

print ('Test set results:')
print ('User mean RMS: %s'%np.mean(err_test_UM))
print ('User mean MAE: %s'%np.mean(err_test_UM_MAE))
print ('\n')
print ('Item mean RMS: %s'%np.mean(err_test_IM))
print ('Item mean MAE: %s'%np.mean(err_test_IM_MAE))
print ('\n')
print ('Global mean RMS: %s'%np.mean(err_test_GM))
print ('Global mean MAE: %s'%np.mean(err_test_GM_MAE))
print ('\n')

print ('Train set results:')
print ('User mean RMS: %s'%np.mean(err_train_UM))
print ('User mean MAE: %s'%np.mean(err_train_UM_MAE))
print ('\n')
print ('Item mean RMS: %s'%np.mean(err_train_IM))
print ('Item mean MAE: %s'%np.mean(err_train_IM_MAE))
print ('\n')
print ('Global mean RMS: %s'%np.mean(err_train_GM))
print ('Global mean MAE: %s'%np.mean(err_train_GM_MAE))
print ('\n')

print ('Linear regression results:')
print ('Train set:')
print ('RMS: %s'%np.mean(err_train_LR))
print ('MAE: %s'%np.mean(err_train_LR_MAE))

print ('Test set:')
print ('RMS: %s'%np.mean(err_test_LR))
print ('MAE: %s'%np.mean(err_test_LR_MAE))

print('Linear Regression paramaters:')
print('alpha = {}'.format(np.mean(alpha)))
print('beta = {}'.format(np.mean(beta)))
print('gamma = {}'.format(np.mean(gamma)))






