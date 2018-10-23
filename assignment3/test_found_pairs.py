import sys
import numpy as np

sig_length = sys.argv[1]
seed = sys.argv[2]

print ('Checking results for sig_length=%s, seed=%s'%(sig_length,seed))

all_sim_pairs =  np.loadtxt('./all_similar_pairs.txt',delimiter=',')
found_pairs = np.loadtxt('./results_%s_%s.txt'%(sig_length,seed),delimiter=',')

number_wrong = 0
number_correct = 0
# list of tuples to check if we've seen a combination before
combinations_seen = []
number_duplicate = 0
number_duplicate_swapped = 0
for i in range(found_pairs.shape[0]):
	user1, user2 = int(found_pairs[i][0]), int(found_pairs[i][1])
	if (user1, user2) not in combinations_seen:
		if (user2, user1) not in combinations_seen:
			combinations_seen.append((user1,user2))
		else:
			print ('Duplicate swapped pair found')
			number_duplicate_swapped += 1
	else:
		print ('Duplicate entry found..')
		number_duplicate += 1

	# find all users2 that this user1 is similar to, and their similarity measure
	# the 'true' values
	sim_users = all_sim_pairs[:,1][ all_sim_pairs[:,0] == user1]
	jaccard = all_sim_pairs[:,2][ all_sim_pairs[:,0] == user1]

	if user2 not in np.asarray(sim_users,dtype=int):
		number_wrong += 1
		print ('Wrong pair found.')
	else:
		real_jaccard = jaccard[ sim_users == user2 ]
		calc_jaccard = found_pairs[i][2]
		# print (real_jaccard, calc_jaccard)
		# print ('Difference between real and calc jaccard:', real_jaccard - calc_jaccard)
		number_correct += 1



print ('Correct number of pairs: ', number_correct)
print ('Wrong number of pairs: ', number_wrong)
print ('Number of duplicate pairs: ', number_duplicate)
print ('Number of duplicate (swapped) pairs: ', number_duplicate_swapped)


