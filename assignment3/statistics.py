import numpy as np
import matplotlib.pyplot as plt


similarity = 0.1
signature_length = 64
num_bands = 16

# Probability users are identical in one part band, given their similarity
def users_identical_one_particular_band(similarity,signature_length,num_bands):
	r = signature_length/num_bands # amount of integers per band

	return similarity**r

print ('Probability U1, U2 are identical in one band:')
print (users_identical_one_particular_band(similarity,signature_length,num_bands))

def users_not_identical_in_any_band(similarity,signature_length,num_bands):
	prob_one_band = users_identical_one_particular_band(similarity,signature_length,num_bands)
	return ( 1 - prob_one_band )**num_bands

print ('Probability U1, U2 are not identical in any of the bands')
print (users_not_identical_in_any_band(similarity,signature_length,num_bands))

def users_identical_in_ATLEAST_one_band(similarity,signature_length,num_bands):

	return 1 - users_not_identical_in_any_band(similarity,signature_length,num_bands)

print ('Probability U1, U2 are identical in AT LEAST one band')
print (users_identical_in_ATLEAST_one_band(similarity,signature_length,num_bands))

def users_identical_in_ATLEAST_two_bands(similarity,signature_length,num_bands):

	return (users_identical_in_ATLEAST_one_band(similarity,signature_length,num_bands)
			- users_identical_one_particular_band(similarity,signature_length,num_bands)
			* users_not_identical_in_any_band(similarity,signature_length,num_bands-1)
			* num_bands )

print ('Probability U1, U2 are identical in AT LEAST two bands')
print (users_identical_in_ATLEAST_two_bands(similarity,signature_length,num_bands))


def calc_probabilities_minus(similarity1, similarity2):
	
	assert similarity1 > similarity2

	probabilities1 = np.empty((101,50))
	probabilities2 = np.empty((101,50))

	for i, signature_length in enumerate(range(50,151)):
		for j, num_bands in enumerate(range(2,51)):
			probabilities1[i,j] = (users_identical_in_ATLEAST_one_band(similarity1,signature_length,num_bands)
								- users_identical_in_ATLEAST_one_band(similarity2,signature_length,num_bands) )
			probabilities2[i,j] = (users_identical_in_ATLEAST_two_bands(similarity1,signature_length,num_bands)
								- users_identical_in_ATLEAST_two_bands(similarity2,signature_length,num_bands) )


	return probabilities1, probabilities2

def calc_minus_probabilities():
	# Calculates (the prob we find U1,U2 in at least 1(and 2) bands 
	# as function of signature length and number of bands
	# assuming U1,U2 are 50 percent similar)
	# minus ( the prob we find U1,U2 in at least 1(and 2) bands 
	# as function of signature length and number of bands
	# assuming U1,U2 are 10 percent similar))

	probabilities1, probabilities2 = calc_probabilities_minus(0.5,0.1)

	x = range(2,51)
	y = range(50,151)
	# x and y are bounds, so z should be the value *inside* those bounds.
	# Therefore, remove the last value from the z array.
	plt.pcolor(x, y, probabilities1[:-1,:-1])
	plt.title('Probability 0.5 similar in at least one band minus 0.1 similar')
	plt.colorbar()
	plt.xlabel('Number of bands')
	plt.ylabel('Signature length')
	plt.savefig('./probabilities1_05minus01')
	plt.show()
	plt.close()

	plt.pcolor(x, y, probabilities2[:-1,:-1])
	plt.title('Probability similar in at least two band')
	plt.colorbar()
	plt.xlabel('Number of bands (not real numbers)')
	plt.ylabel('Signature length (not real numbers)')
	plt.savefig('./probabilities2_05minus01')
	plt.show()
	plt.close()

calc_minus_probabilities()

def calc_probabilities():	
	# Calculates (the prob we find U1,U2 in at least 1(and 2) bands 
	# as function of signature length and number of bands
	# assuming U1,U2 are 50 percent similar)

	probabilities1 = np.empty((101,50))
	probabilities2 = np.empty((101,50))
	similarity = 0.5

	for i, signature_length in enumerate(range(50,151)):
		for j, num_bands in enumerate(range(2,51)):
			probabilities1[i,j] = users_identical_in_ATLEAST_one_band(similarity,signature_length,num_bands)
			probabilities2[i,j] = users_identical_in_ATLEAST_two_bands(similarity,signature_length,num_bands)

	x = range(2,51)
	y = range(50,151)
	# x and y are bounds, so z should be the value *inside* those bounds.
	# Therefore, remove the last value from the z array.
	plt.pcolor(x, y, probabilities1[:-1,:-1])
	plt.title('Probability 0.5 similar in at least one band')
	plt.colorbar()
	plt.xlabel('Number of bands')
	plt.ylabel('Signature length')
	plt.savefig('./probabilities1_05')
	plt.show()
	plt.close()

	plt.pcolor(x, y, probabilities2[:-1,:-1])
	plt.title('Probability 0.5 similar in at least two band')
	plt.colorbar()
	plt.xlabel('Number of bands')
	plt.ylabel('Signature length')
	plt.savefig('./probabilities2_05')
	plt.show()
	plt.close()
	
calc_probabilities()

