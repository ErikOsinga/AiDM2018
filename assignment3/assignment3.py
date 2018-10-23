import sys
import numpy as np
from scipy.sparse import csc_matrix
from collections import defaultdict
import itertools

print ('Setting the random seed to:', sys.argv[1])
np.random.seed(int(sys.argv[1]))

print("Loading data from file: ", sys.argv[2])
file = sys.argv[2]
pairs = np.load(file)

users = pairs[:,0]
movies = pairs[:,1]

def create_sparse_matrix():
	data = np.ones(len(users)) # Create 65M ones
	# Construct the sparse matrix containing ones where (user,movie) is rated
	X = csc_matrix((data, (users, movies)), dtype = np.int8)

	return X

def find_signatures(M, X, cur_n): 
	"""
	M = signature matrix (users x signatures)
	X = csc sparse array of size (users x movies)
	cur_n = current iteration in len(signatures)
	testlength = the amount of columns we inspect for the first nonzero argument
	"""
	testlength = 600
	# we expect the first 1 to occur with very high certainty in the first 
	# 600 columns, because every user has rated atleast 300 movies.
	possible_signature = np.asarray(X[:,:testlength].argmax(axis=1))
	# Find the small subset of users for which this (possibly) didnt work.
	# It did work if those users actually contain a 1 in the first column
	# Mask = the users for which it didnt work 
	mask = ( (possible_signature == 0)[:,0] & (X[:,0].toarray() == 0)[:,0] )
	if np.sum(mask) != 0:
		# print ('For %i users we did not find the first nonzero'%np.sum(mask))
		possible_signature[mask] = np.asarray(X[mask,:].argmax(axis=1))
	M[:,cur_n] = possible_signature[:,0]
			
def create_signatures(X, num_sig):
	print("Creating signatures...")
	M = np.zeros((X.shape[0],num_sig)) #signature matrix (users,signatures)

	shuff_list = np.arange(X.shape[1])
	for i in range(num_sig): # create num_sig permutations 

		np.random.shuffle(shuff_list)
		X = X[:,shuff_list]

		find_signatures(M, X, i)

	return M

# break signatures into b bands
def partition_into_bands(M, b):
	print("Partitioning into bands...")
	# list of bands
	A = np.split(M, b, axis=1) # can only split into M.shape[1]/b = integer bands
	# represent each band by a tuple
	list_of_buckets = [] # we want a dictionary of signature:[users] for each band
	for band in range(len(A)):
		buckets = defaultdict(list)
		for user in range(A[band].shape[0]):
			tup = tuple(A[band][user]) # the part of the signature in this band
			buckets[tup].append(user)
		# If a bucket in this band only contains 1 user, delete it
		for i in list(buckets.keys()):
			if len(buckets[i]) == 1:
				del buckets[i]

		list_of_buckets.append(buckets) # list of 'band' number of buckets 

	return list_of_buckets

def unique_user_pairs(list_of_buckets, sig_len):
	unique_pairs = set()
	unique_pairs2 = set()
	print("Finding unique candidate pairs...")
	for i in range(len(list_of_buckets)):
		for bucket in list_of_buckets[i].keys():
			if len(list_of_buckets[i][bucket]) < 1000:
				all_pairs = set(pair for pair in itertools.combinations(list_of_buckets[i][bucket], 2))
				all_pairs = all_pairs.difference(unique_pairs) # Check if we dont already have this pair
				for test_pair in all_pairs:
					# Python3+ has automatic float division, no casting required
					sim = np.count_nonzero(M[test_pair[0]] == M[test_pair[1]])/sig_len
					if sim > 0.5:
						unique_pairs.add(test_pair)
#			else:
#				all_pairs = set(pair for pair in itertools.combinations(list_of_buckets[i][bucket][:1000], 2))
#				all_pairs = all_pairs.difference(unique_pairs)
#				for test_pair in all_pairs:
#					sim = float(np.count_nonzero(M[test_pair[0]] == M[test_pair[1]]))/sig_len
#					if sim > 0.5:
#						unique_pairs.add(test_pair)
	return unique_pairs

def jaccard_calculation(unique_pairs, X):
	print("Calculating Jaccard Similarity...")
	X_array = X.A
	F = open('./results.txt','w')
	for user1, user2 in unique_pairs:
		intersection = np.sum(X_array[user1, :] & X_array[user2, :])
		union = np.sum(X_array[user1, :] | X_array[user2, :])
		jaccard_sim = intersection/union # again no casting to float necessary
		if jaccard_sim >= 0.5:
			F.write('%s,%s,%s\n'%(user1,user2,jaccard_sim))
	F.close()

def check_extra_overlap():
	X_array = X.A
	# We check all possible combinations of users that we have found in results.txt
	all_sim = np.loadtxt('./results.txt', delimiter = ',', usecols=(0,1))
	all_set = set(tuple(pair) for pair in all_sim)
	unique_elements = np.unique(all_sim)
	pairs = set(pair for pair in itertools.combinations(unique_elements, 2))
	# Find the users we have not yet compared
	pairs = pairs.difference(all_set)
	F = open('./results.txt', 'a')
	for test_pair in pairs:
		if test_pair not in all_sim:
			intersection = np.sum(X_array[int(test_pair[0]), :] & X_array[int(test_pair[1]), :])
			union = np.sum(X_array[int(test_pair[0]), :] | X_array[int(test_pair[1]), :])
			jaccard_sim = intersection/union
			if jaccard_sim >= 0.5:
				print (test_pair)
				F.write('%s,%s,%s\n'%(int(test_pair[0]),int(test_pair[1]),jaccard_sim))
	F.close()

#Set parameters:
sig_len = 92 #length of signature
b = 23 #number of bands

#Create a sparse matrix of (users x movies):
X = create_sparse_matrix()

#Create signatures of length sig_len:
M = create_signatures(X, sig_len)

#Hash signatures into buckets using b bands:
list_of_buckets = partition_into_bands(M,b)

#Find the unique user pairs which are candidates for being similar
unique = unique_user_pairs(list_of_buckets, sig_len)

#Calculate the actual similarity of the candidate pairs and write to a file
sim_users = jaccard_calculation(unique, X)

check_extra_overlap()