import sys
import numpy as np
from scipy.sparse import csc_matrix, hstack
from collections import defaultdict



pairs = np.load('./user_movie.npy')
print ('Setting the random seed to:', sys.argv[1])
np.random.seed(int(sys.argv[1]))

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
   return M

def find_signatures2(M, X, cur_n):
   temp = np.array([X.indices[X.indptr[j]:X.indptr[j+1]] for j in range(X.shape[1])])
   k=0
   while np.isnan(M[:,cur_n]).any():
      for l in temp[k]:
         if np.isnan(M[l,cur_n]):
            M[l,cur_n] = k
      k += 1
   return M
            
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
		A = [X.getcol(j) for j in shuff_list]
		X = hstack(A)
		# keep in mind, maybe permuting the matrix is quicker than creating a new
#		X = csr_matrix((X.data, (users, movies)))
        
		# save the indices of the first nonzero of every user for this permutation
		#M[:,i] = X.argmax(axis=1) # have to find a better way to find first nonzero\
		
		#use find_signatures function
		M = find_signatures(M, X.tocsr(), i)

	return M

# M = create_signatures(X, 50)
M = np.load('./signature_matrix.npy') # for testing purposes

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

b = 10 # more bands makes it slower
list_of_buckets, user_dict = partition_into_bands(M,b)

def check_buckets(list_of_buckets, user_dict):
	"""
	Check how many users are in each bucket in each band. Save only the ones 
	that have multiple users in a buckets in multiple bands
	"""

	# for buckets in range(len(list_of_buckets)):
	# 	for key in list_of_buckets[buckets].keys():
	# 		# inspect this particular bucket in this band
	# 		if len(buckets[key]) < 2:
	# 			buckets.pop(key, None) # remove this key from dict as it is not interesting

	# for buckets in range(len(list_of_buckets)):
	# 	for key in list_of_buckets[buckets].keys():
	# 		for user in list_of_buckets[buckets][key]:
	# 			user_dict[user]

	for i, buckets in enumerate(list_of_buckets): 
		# buckets is the dictionary with signatures as keys and users as values
		# from this we ask, what users is 'user' sharing this bucket with
		buckets[user_dict[user][i]]
	for user in user_dict.keys():
		#list with all users which are in the same bucket as a specific user in any band
		buckets_list = [list_of_buckets[i][user_dict[user][i]] for i in range(len(list_of_buckets))] 
		# flatten it to find unique elements
 		buckets_list = np.hstack(buckets_list)
		(users, counts) = np.unique(buckets_list,return_counts=True)
		users = users[counts > 1] # [current user, any users that it shares > 1 bucket with]
		





    
