import sys
import numpy as np
from scipy.sparse import csc_matrix, hstack



pairs = np.load('./user_movie.npy')
print ('Setting the random seed to:', sys.argv[1])
np.random.seed(int(sys.argv[1]))

print ('Testing with first 1e6 user,movie pairs')
users = pairs[:int(1e6),0]
movies = pairs[:int(1e6),1]

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

preload = False

if preload: 
	M = np.load('./signature_matrix.npy')
else:
	M = create_signatures(X, 50)
	np.save('./signature_matrix1e6.npy',M)

def test_checking_keys(M, b):
	A = np.split(M, b, axis=1) # can only split into M.shape[1]/b = integer bands

	list_of_buckets = []
	for band in range(len(A)):
		buckets = dict()
		for user in range(A[band].shape[0]):
			print (band, user)
			tup = tuple(A[band][user])
			# check if dict already contains the signature as a key
			if tup in buckets.keys():
				buckets[tup].append(user)
			else: # first time finding this key
				buckets[tup] = [user]
		list_of_buckets.append(buckets) # list of 'band' number of buckets 	
	
	return list_of_buckets

list_of_buckets = test_checking_keys(M,10)

# 1000 loops, best of 3: 499 micros per loop (for using += and not append)
# 1000 loops, best of 3: 507 micros per loop ( for using append)
# 1000 loops, best of 3: 364 micros per loop for not using the if statement