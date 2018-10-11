import sys
import numpy as np
from scipy.sparse import coo_matrix


pairs = np.load('./user_movie.npy')
print ('Setting the random seed to:', sys.argv[1])
np.random.seed(int(sys.argv[1]))

users = pairs[:,0]
movies = pairs[:,1]

def create_sparse_matrix2():
	data = np.ones(len(users)) # create 65M ones
	# construct the sparse matrix containing ones where (user,movie) is rated
	X = coo_matrix((data, (users, movies)))

	return X

X = create_sparse_matrix2()

def find_signatures(M, X, cur_n):   
   M[:,cur_n] = [X.indices[X.indptr[j]:X.indptr[j+1]][0] for j in range(X.shape[0])]
   return M


def create_signatures(X, num_sig):
	# signature matrix
	M = np.zeros((X.shape[0],num_sig)) 
	M.fill(np.nan)
	t_movies = movies
	length = len(t_movies)
	s = X.shape[0]
	for i in range(num_sig): # create num_sig permutations 
		print(i)
		#np.random.shuffle(movies) # permute the movies
		#alternative to shuffle:
		t_movies = np.mod(t_movies + np.random.randint(s, size=length),s)
		# keep in mind, maybe permuting the matrix is quicker than creating a new
		X = coo_matrix(X.data, (users, t_movies))
		# save the indices of the first nonzero of every user for this permutation
		#M[:,i] = X.argmax(axis=1) # have to find a better way to find first nonzero\
		
		#use find_sigantures function
		M = find_signatures(M, X.tocsr(), i)

	return M
M = create_signatures(X, 50)
    
