import numpy as np
from sklearn import linear_model

# Array of shape (1000209,4) containing
# UserID::MovieID::Rating::Timestamp
ratings = np.loadtxt('./ml-1m/ratings.dat',delimiter="::")

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

# Calculate all user means and item means into a dictionary
# For quick accessing
all_users = np.unique(ratings[:,0])
all_items = np.unique(ratings[:,1])    
user_means = dict()
item_means = dict()
for user in all_users:
	user_means[user] = user_mean(ratings,user)
for item in all_items:
	item_means[item] = item_mean(ratings,item)


def five_fold_CV(ratings):
	'''Test the linear regression model with 5-fold CV '''
	np.random.seed(17) # For reproducibility
   
   #split data into 5 train and test folds
	nfolds=5

   #allocate memory for results:
	err_train=np.zeros(nfolds)
	err_test=np.zeros(nfolds)
	err_test_UM=np.zeros(nfolds)
   
	seqs=[x%nfolds for x in range(len(ratings))]
	np.random.shuffle(seqs)

   #for each fold:
	for fold in range(nfolds):
		print ('Fold number: %i'%fold)
       
		train_sel=np.array([x!=fold for x in seqs])
		test_sel=np.array([x==fold for x in seqs])
		train=ratings[train_sel]
		test=ratings[test_sel]
       
		alpha, beta, gamma = train_linear_combination(train)
		err_train[fold] = np.mean((train[:,2] - predict_all_ratings(train,alpha,beta,gamma) )**2)
		err_test[fold] = np.mean((test[:,2] - predict_all_ratings(test,alpha,beta,gamma) )**2)

		train_items = np.unique(train[:,1]) 
		train_users = np.unique(train[:,0]) 
		train_user_means = dict()
		train_item_means = dict()
		for user in train_users:
			train_user_means[user] = user_mean(train,user)
		for item in train_items:
			 train_item_means[item] = item_mean(train,item)
		err_t = []  
		for line in range(len(test)):
			if test[line,0] in train_users:
				err_t.append( (test[line,2] - train_user_means[test[line,0]])**2  )
			else: print("continue")
		print(len(test))
		err_test_UM[fold] = np.mean(err_t) 
		print(err_test_UM)  
    
          
	return err_train, err_test, err_test_UM
       
def train_linear_combination(train):

   # Find all user means and all item means
	y = train[:,2]
   # construct the x1,x2 matrix
	x1 = []
	x2 = []

	for row in range(len(train)):
		user = train[row,0]
		item = train[row,1]
		x1.append(user_means[user])
		x2.append(item_means[item])
      
	X = np.asarray([x1,x2]).T
	reg = linear_model.LinearRegression()
	reg.fit(X,y)
	alpha, beta = reg.coef_
	gamma = reg.intercept_
   
	return alpha,beta,gamma

def predict_all_ratings(ratings,alpha,beta,gamma):
	all_ratings = []
	for row in range(len(ratings)):
		user = ratings[row,0]
		item = ratings[row,1]
		all_ratings.append(predict_linear_combination(ratings,user,item,alpha,beta,gamma))
   
	return np.asarray(all_ratings)
  
def predict_linear_combination(ratings,user,item,alpha,beta,gamma):
	prediction = (alpha * user_means[user] 
               + beta*item_means[item] + gamma)
	return prediction
      
   
#alpha,beta,gamma = train_linear_combination(ratings)
# for quick testing purposes
alpha = 0.7811833
beta = 0.87454303
gamma = -2.3489102231862939

err_train, err_test, err_test_UM= five_fold_CV(ratings)









