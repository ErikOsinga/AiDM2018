import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt

# parameter_tests =  np.loadtxt('./parameter_tests.txt',delimiter=',')


parameter_tests = ascii.read('./parameter_tests.txt',format='csv')

group_siglength = parameter_tests.group_by('num_signatures')

siglength64 = group_siglength.groups[0]
siglength90 = group_siglength.groups[1]
siglength92 = group_siglength.groups[2]
siglength100 = group_siglength.groups[3]
siglength150 = group_siglength.groups[4]




def geef_statistics(signlength):
	signlength_signs = signlength.group_by('sign_threshold')

	# print (len(signlength.groups))
	i = 1
	cur_table = signlength_signs.groups[i]

	sign_length = cur_table['num_signatures'][0]
	sign_threshold = cur_table['sign_threshold'][0]
	average_time = np.mean(cur_table['time'])
	std_time = np.std(cur_table['time'])
	average_num_users = np.mean(cur_table['count_number_users'])
	std_num_users = np.std(cur_table['count_number_users'])

	pair_per_minute = cur_table['count_number_users'] / cur_table['time'] * 60
	average_pairs_min = np.mean(pair_per_minute)
	std_pairs_min = np.std(pair_per_minute)

	print ('Result for %i length signatures:'%sign_length)
	print ('Signature threshold: ', sign_threshold)
	# print ('Average time: %.3f seconds = %.3f minutes'%(average_time,average_time/60.))
	# print ('Std time: %.3f seconds = %.3f minutes'%(std_time,std_time/60.))
	# print ('Average number of users found: %.2f'%average_num_users)
	print ('Average Pairs per minute: %.2f'%average_pairs_min)
	print ('Standard deviation: %.2f'%std_pairs_min)
	print ('\n')


# geef_statistics(siglength64)
# geef_statistics(siglength90)
# geef_statistics(siglength92)
# geef_statistics(siglength100)
# geef_statistics(siglength150)

parameter_tests2 = ascii.read('./parameter_tests2.txt',format='csv')

def geef_statistics2(parameter_tests):
	group_siglength = parameter_tests.group_by('num_signatures')

	for i in range(2):
		group_i = group_siglength.groups[i]
		signlength = group_i['num_signatures'][0]

		pair_per_minute = group_i['count_number_users'] / group_i['time'] * 60
		average_pairs_min = np.mean(pair_per_minute)
		std_pairs_min = np.std(pair_per_minute)

		print ('Result for %i length signatures:'%signlength)
		print ('Average Pairs per minute: %.2f'%average_pairs_min)
		print ('Standard deviation: %.2f'%std_pairs_min)
		print ('\n')

		plt.hist(pair_per_minute)
		plt.title('Result for signature length %i:'%signlength)
		plt.xlabel('Pairs per minute')
		plt.ylabel('Count')
		plt.show()

geef_statistics2(parameter_tests2)