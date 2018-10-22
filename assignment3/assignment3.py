import sys
import numpy as np
from scipy.sparse import csc_matrix
from collections import defaultdict
import itertools

print ('Setting the random seed to:', sys.argv[1])
np.random.seed(int(sys.argv[1]))

print("Loading data...")

pairs = np.load('./user_movie.npy')

users = pairs[:,0]
movies = pairs[:,1]

def create_sparse_matrix():
	data = np.ones(len(users)) # create 65M ones
	# construct the sparse matrix containing ones where (user,movie) is rated
	X = csc_matrix((data, (users, movies)), dtype = np.int8)

	return X

X = create_sparse_matrix()

def find_signatures(M, X, cur_n):   
	M[:,cur_n] = [X.indices[X.indptr[j]:X.indptr[j+1]][0] for j in range(X.shape[0])]



def find_signatures(M, X, cur_n, testlength): 
	"""
	M = signature matrix (users x signatures)
	X = sparse array of size (users x movies)
	cur_n = current iteration in len(signatures)
	testlength = the amount of columns we inspect for the first nonzero argument
	"""
	# we expect the first 1 to occur with very high certainty in the first 
	# 600 columns, because every user has rated atleast 300 movies 
	possible_signature = np.asarray(X[:,:testlength].argmax(axis=1))
	# find the small subset of users for which this (possibly) didnt work
	# it did work if those users actually contain a 1 in the first column
	# so we mask these users out of the mask
	mask = ( (possible_signature == 0)[:,0] & (X[:,0].toarray() == 0)[:,0] )
	if np.sum(mask) != 0:
		# print ('For %i users we did not find the first nonzero'%np.sum(mask))
		possible_signature[mask] = np.asarray(X[mask,:].argmax(axis=1))
	M[:,cur_n] = possible_signature[:,0]
			
def create_signatures(X, num_sig, testlength):
	print("Creating signatures...")
	M = np.zeros((X.shape[0],num_sig)) #signature matrix (users,signatures)

	shuff_list = np.arange(X.shape[1])
	for i in range(num_sig): # create num_sig permutations 

		np.random.shuffle(shuff_list)
		X = X[:,shuff_list]

		find_signatures(M, X, i, testlength)

	return M

M = create_signatures(X, 64, 600)

# break signatures into b bands
def partition_into_bands(M, b):
	print("Partitioning into bands...")
	# list of bands
	A = np.split(M, b, axis=1) # can only split into M.shape[1]/b = integer bands
								# or use np.vsplit
	# represent each band by a tuple
	list_of_buckets = [] # we want a dictionary of signature:[users] for each band
	for band in range(len(A)):
		buckets = defaultdict(list)
		for user in range(A[band].shape[0]):
			tup = tuple(A[band][user]) # the part of the signature in this band
			buckets[tup].append(user)
		for i in list(buckets.keys()):
			if len(buckets[i]) == 1:
				del buckets[i]

		list_of_buckets.append(buckets) # list of 'band' number of buckets 

	return list_of_buckets#, user_dict

b = 16 # more bands makes it slower
list_of_buckets = partition_into_bands(M,b)

def unique_user_pairs(list_of_buckets):
	unique_pairs = set()
	print("Finding unique pairs...")
	for i in range(len(list_of_buckets)):
		for bucket in list_of_buckets[i].keys():
			all_pairs = set(pair for pair in itertools.combinations(list_of_buckets[i][bucket], 2))
			all_pairs = all_pairs.difference(unique_pairs)
			for test_pair in all_pairs:
				sim = float(np.count_nonzero(M[test_pair[0]] == M[test_pair[1]]))/64.
				if sim > 0.5:
					unique_pairs.add(test_pair)
	return unique_pairs
unique = unique_user_pairs(list_of_buckets)

def jaccard_calculation(unique_pairs, X):
	print("Calculating Jaccard Similarity...")
	X_array = X.A
	F = open('./results.txt','w')
	for user1, user2 in unique_pairs:
		intersection = np.sum(X_array[user1, :] & X_array[user2, :])
		union = np.sum(X_array[user1, :] | X_array[user2, :])
		jaccard_sim = float(intersection)/float(union)
		if jaccard_sim >= 0.5:
			F = open('./results.txt', 'a')
			F.write('%s,%s,%s\n'%(user1,user2,jaccard_sim))
			F.close()

sim_users = jaccard_calculation(unique, X)