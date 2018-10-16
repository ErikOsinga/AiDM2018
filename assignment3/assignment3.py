import sys
import numpy as np
from scipy.sparse import csc_matrix, hstack
from collections import defaultdict
from sklearn.metrics import jaccard_similarity_score

print ('Setting the random seed to:', sys.argv[1])
np.random.seed(int(sys.argv[1]))

pairs = np.load('./user_movie.npy')

users = pairs[:,0]
movies = pairs[:,1]

def create_sparse_matrix2():
	data = np.ones(len(users)) # create 65M ones
	# construct the sparse matrix containing ones where (user,movie) is rated
	X = csc_matrix((data, (users, movies)))

	return X

X = create_sparse_matrix2()

def find_signatures(M, X, cur_n):   
   M[:,cur_n] = [X.indices[X.indptr[j]:X.indptr[j+1]][0] for j in range(X.shape[0])]


def find_signatures2(M, X, cur_n):
   temp = np.array([X.indices[X.indptr[j]:X.indptr[j+1]] for j in range(X.shape[1])])
   k=0
   while np.isnan(M[:,cur_n]).any():
      for l in temp[k]:
         if np.isnan(M[l,cur_n]):
            M[l,cur_n] = k
      k += 1

            
def create_signatures(X, num_sig):
	# signature matrix
	M = np.zeros((X.shape[0],num_sig)) #(users,signatures)
	M.fill(np.nan)
	shuff_list = np.arange(X.shape[1])
	for i in range(num_sig): # create num_sig permutations 
		print(i)
#		np.random.shuffle(movies) # permute the movies

		#alternative to shuffle:
		np.random.shuffle(shuff_list)
		X = X[:,shuff_list]

#		X = csr_matrix((X.data, (users, movies)))
        
		# save the indices of the first nonzero of every user for this permutation
		#M[:,i] = X.argmax(axis=1) # have to find a better way to find first nonzero\
		
		#use find_signatures function
		find_signatures(M, X.tocsr(), i)

	return M

#M = create_signatures(X, 64)
M = np.load('./signature_matrix.npy') # for testing purposes (50 signatures)

# break signatures into b bands
def partition_into_bands(M, b):

	# list of bands
	A = np.split(M, b, axis=1) # can only split into M.shape[1]/b = integer bands
								# or use np.vsplit
	# represent each band by a tuple
	user_dict = defaultdict(list) # dictionary with the users as keys and buckets as values
	list_of_buckets = [] # we want a dictionary of signature:[users] for each band
	for band in range(len(A)):
		buckets = defaultdict(list)
		for user in range(A[band].shape[0]):
			tup = tuple(A[band][user]) # the part of the signature in this band

			user_dict[user].append(tup)
			buckets[tup].append(user)

		list_of_buckets.append(buckets) # list of 'band' number of buckets 

	return list_of_buckets, user_dict

b = 16 # more bands makes it slower
list_of_buckets, user_dict = partition_into_bands(M,b)

def check_buckets(list_of_buckets, user_dict, sharing_buckets):
	"""
	Check how many users are in each bucket in each band. Save only the ones 
	that have multiple users in a bucket in 'sharing_buckets' amount of bands 
	"""
	overlap_users = dict() #dictionary with key:user, values:list of all users with which it shares >1 bucket

	for user in user_dict.keys():

		#list with all users which are in the same bucket as a specific user in any band
		buckets_list = [list_of_buckets[i][user_dict[user][i]] for i in range(len(list_of_buckets))] 
		# flatten it to find unique elements
 		buckets_list = np.hstack(buckets_list)
		(users, counts) = np.unique(buckets_list,return_counts=True)
		users = users[counts > sharing_buckets] # [current user, any users that it shares > 1 bucket with]
		if len(users) > 1:
			overlap_users[users[0]] = users[1:]
			# misschien dat we hier moeten checken op users die al geweest zijn
			# bijv 0,5 .. als we dan bij user 5 komen hoeft die niet weer met 0 
			# maar dat doen we nu niet.
	return overlap_users

overlap_users = check_buckets(list_of_buckets, user_dict, sharing_buckets=2) 

def calculate_similarity(overlap_users, X):
	# for the probable similar users, calculate the Jaccard Similarity
	# Defined as Sim(U1,U2) = |C1 <intersection> C2| / |C1 <union> C2|

	# either use sklearn.metrics.jaccard_similarity_score 
	# or 1 - scipy.spatial.distance.jaccard
	print ('Going to check overlap users now:', len(overlap_users))
	F = open('./results.txt','w')
	for user, values in overlap_users.items():
		for overlap_user in values: 
			if user < overlap_user:
				print user
				# calculate the similarity for this user, overlap_user pair
				if jaccard_similarity_score(X[user], X[overlap_user]) >= 0.5:
					F = open('./results.txt','a')
					F.write('%s,%s\n'%(user,overlap_user))
					F.close()
					# temporary for checking purposes
					# B = open('./results_test.txt','a')
					# B.write('%s,%s,%s\n'%(user,overlap_user,jaccard_similarity_score(X[user], X[overlap_user])))
					# B.close()

calculate_similarity(overlap_users,X)

# 60 signatures and 15 bands