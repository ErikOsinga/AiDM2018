import numpy as np
import matplotlib.pyplot as plt

def plot_params(ax,xlabel,ylabel,title=''):
	ax.get_xaxis().tick_bottom()    
	ax.get_yaxis().tick_left()  
	ax.spines["top"].set_visible(False)    
	ax.spines["bottom"].set_visible(False)    
	ax.spines["right"].set_visible(False)    
	ax.spines["left"].set_visible(False) 
	plt.title(title,fontsize=16)
	ax.set_xlabel(xlabel,fontsize=18)
	ax.set_ylabel(ylabel,fontsize=18)
	ax.legend(fontsize=18, framealpha=0, loc=1)
	plt.tight_layout()
	ax.tick_params(axis="both", which="both", bottom=False, top=False,    
                labelbottom=True, left=False, right=False, labelleft=True) 


def plot_error_matrix_factorization():
	'''
	Plots the error as a function of iteration 
	for a single (1/5) fold of matrix factorization
	'''

	# Loading results
	all_RMSE_train = np.load('./data/RMSE_test_it.npy')
	all_RMSE_test = np.load('./data/RMSE_train_it.npy')
	all_MAE_train = np.load('./data/MAE_train_it.npy')
	all_MAE_test = np.load('./data/MAE_test_it.npy')

	# plot RMSE train and test
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.plot(all_RMSE_train,label='Training set')
	ax.plot(all_RMSE_test,label='Test set')
	for y in np.arange(0.6, 2.0, 0.2):    
		ax.plot(range(0, 75), [y] * len(range(0, 75)), "--", lw=0.5, color="black", alpha=0.3)   
	plot_params(ax,'Iteration','RMSE','Matrix Factorization')
	plt.ylim(0.5,2.0)
	plt.savefig("MF_RMSE.pdf")
	plt.show()

	# plot MAE train and test
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.plot(all_MAE_train,label='Training set')
	ax.plot(all_MAE_test,label='Test set')
	for y in np.arange(0.6, 2.0, 0.2):    
		ax.plot(range(0, 75), [y] * len(range(0, 75)), "--", lw=0.5, color="black", alpha=0.3)   	
	plot_params(ax,'Iteration','MAE','Matrix Factorization')
	plt.ylim(0.5,2.0)
	plt.savefig("MF_MAE.pdf")
	plt.show()

def make_table_matrix_factorization():
	# if not normalization:
	F = open('./data/matrix_factorization_normalization.csv','w')
	all_U = np.load('./data/all_U.npy') # shape (5, 6040, 10)
	all_M = np.load('./data/all_M.npy') # shape (5, 10, 3706)
	all_RMSE_train = np.load('./data/all_RMSE_train.npy')
	all_RMSE_test = np.load('./data/all_RMSE_test.npy')
	all_MAE_train = np.load('./data/all_MAE_train.npy')
	all_MAE_test = np.load('./data/all_MAE_test.npy')

	all_U_normalization = np.load('./data/all_U_normalization.npy') # shape (5, 6040, 10)
	all_M_normalization = np.load('./data/all_M_normalization.npy') # shape (5, 10, 3706)
	all_RMSE_train_normalization = np.load('./data/all_RMSE_train_normalization.npy')
	all_RMSE_test_normalization = np.load('./data/all_RMSE_test_normalization.npy')
	all_MAE_train_normalization = np.load('./data/all_MAE_train_normalization.npy')
	all_MAE_test_normalization = np.load('./data/all_MAE_test_normalization.npy')

	# old format
	# F.write(',\hfil Train set,,\hfil Test set,\n')
	# F.write(',mean,std,mean,std\n')
	# F.write('RMSE,%.2f,%.4f,%.2f,%.4f \n'%(np.mean(all_RMSE_train),np.std(all_RMSE_train),np.mean(all_RMSE_test),np.std(all_RMSE_test)))
	# F.write('MAE,%.2f,%.4f,%.2f,%.4f \n'%(np.mean(all_MAE_train),np.std(all_MAE_train),np.mean(all_MAE_test),np.std(all_MAE_test)))
	# F.close()

	F.write(',Without Normalization,,With Normalization,\n')
	F.write(',Mean,Standard deviation,Mean,Standard deviation\n')
	F.write('RMSE  (train),%.2f,%.4f,%.2f,%.4f \n'%(np.mean(all_RMSE_train),np.std(all_RMSE_train),np.mean(all_RMSE_train_normalization),np.std(all_RMSE_train_normalization)))
	F.write('RMSE  (test),%.2f,%.4f,%.2f,%.4f \n'%(np.mean(all_RMSE_test),np.std(all_RMSE_test),np.mean(all_RMSE_test_normalization),np.std(all_RMSE_test_normalization)))
	F.write('MAE  (train),%.2f,%.4f,%.2f,%.4f \n'%(np.mean(all_MAE_train),np.std(all_MAE_train),np.mean(all_MAE_train_normalization),np.std(all_MAE_train_normalization)))
	F.write('MAE  (test),%.2f,%.4f,%.2f,%.4f \n'%(np.mean(all_MAE_test),np.std(all_MAE_test),np.mean(all_MAE_test_normalization),np.std(all_MAE_test_normalization)))
	F.close()

def plot_error_ALS():
	'''
	Plot the error as a function of iteration
	for the ALS algorithm
	'''

	all_RMSE_ALS_train = np.load('./data/all_RMSE_ALS_train.npy')
	all_RMSE_ALS_test = np.load('./data/all_RMSE_ALS_test.npy')

	all_MAE_ALS_train = np.load('./data/all_MAE_ALS_train.npy')
	all_MAE_ALS_test = np.load('./data/all_MAE_ALS_test.npy')

	fold = 0

	# plot RMSE train and test
	fig, ax = plt.subplots()
	ax.plot(all_RMSE_ALS_train[fold],label='Training set')
	ax.plot(all_RMSE_ALS_test[fold],label='Training set')
	for y in np.arange(0.6, 2.0, 0.2):    
		ax.plot(range(0, 75), [y] * len(range(0, 75)), "--", lw=0.5, color="black", alpha=0.3)   	
	plot_params(ax,'Iteration','RMSE','Alternating Least Squares')
	plt.ylim(0.5,2.5)
	plt.savefig('./data/RMSE_ALS.pdf')
	plt.show()

	# plot MAE train and test
	fig, ax = plt.subplots()
	ax.plot(all_MAE_ALS_train[fold],label='Training set')
	ax.plot(all_MAE_ALS_test[fold],label='Training set')
	for y in np.arange(0.6, 2.0, 0.2):    
		ax.plot(range(0, 75), [y] * len(range(0, 75)), "--", lw=0.5, color="black", alpha=0.3)   	
	plot_params(ax,'Iteration','MAE','Alternating Least Squares')
	plt.ylim(0.5,2.5)
	plt.savefig('./data/MAE_ALS.pdf')
	plt.show()

def make_table_ALS():

	F = open('./data/ALS_normalization.csv','w')

 	# shape (5,75 (or less if early stop) )
	all_RMSE_ALS_train = np.load('./data/all_RMSE_ALS_train.npy')
	all_RMSE_ALS_test = np.load('./data/all_RMSE_ALS_test.npy')

	all_MAE_ALS_train = np.load('./data/all_MAE_ALS_train.npy')
	all_MAE_ALS_test = np.load('./data/all_MAE_ALS_test.npy')

 	# shape (5,75 (or less if early stop) )
	all_RMSE_ALS_train_normalization = np.load('./data/all_RMSE_ALS_train_normalization.npy')
	all_RMSE_ALS_test_normalization = np.load('./data/all_RMSE_ALS_test_normalization.npy')

	all_MAE_ALS_train_normalization = np.load('./data/all_MAE_ALS_train_normalization.npy')
	all_MAE_ALS_test_normalization = np.load('./data/all_MAE_ALS_test_normalization.npy')

	final_RMSE_ALS_train = []
	final_RMSE_ALS_test = []
	final_MAE_ALS_train = []
	final_MAE_ALS_test = []

	final_RMSE_ALS_train_normalization = []
	final_RMSE_ALS_test_normalization = []
	final_MAE_ALS_train_normalization = []
	final_MAE_ALS_test_normalization = []

	for fold in range(5):
		final_RMSE_ALS_train.append(all_RMSE_ALS_train[fold][-1])
		final_RMSE_ALS_test.append(all_RMSE_ALS_test[fold][-1])
		final_MAE_ALS_train.append(all_MAE_ALS_train[fold][-1])
		final_MAE_ALS_test.append(all_MAE_ALS_test[fold][-1])

		final_RMSE_ALS_train_normalization.append(all_RMSE_ALS_train_normalization[fold][-1])
		final_RMSE_ALS_test_normalization.append(all_RMSE_ALS_test_normalization[fold][-1])
		final_MAE_ALS_train_normalization.append(all_MAE_ALS_train_normalization[fold][-1])
		final_MAE_ALS_test_normalization.append(all_MAE_ALS_test_normalization[fold][-1])
	
	F.write(',Without Normalization,,With Normalization,\n')
	F.write(',Mean,Standard deviation,Mean,Standard deviation\n')
	F.write('RMSE  (train),%.2f,%.4f,%.2f,%.4f \n'%(np.mean(final_RMSE_ALS_train),np.std(final_RMSE_ALS_train),np.mean(final_RMSE_ALS_train),np.std(final_RMSE_ALS_train)))
	F.write('RMSE  (test),%.2f,%.4f,%.2f,%.4f \n'%(np.mean(final_RMSE_ALS_test),np.std(final_RMSE_ALS_test),np.mean(final_RMSE_ALS_test_normalization),np.std(final_RMSE_ALS_test_normalization)))
	F.write('MAE  (train),%.2f,%.4f,%.2f,%.4f \n'%(np.mean(final_MAE_ALS_train),np.std(final_MAE_ALS_train),np.mean(final_MAE_ALS_train_normalization),np.std(final_MAE_ALS_train_normalization)))
	F.write('MAE  (test),%.2f,%.4f,%.2f,%.4f \n'%(np.mean(final_MAE_ALS_test),np.std(final_MAE_ALS_test),np.mean(final_MAE_ALS_test_normalization),np.std(final_MAE_ALS_test_normalization)))
	
	F.close()


# plot_error_matrix_factorization()
#make_table_matrix_factorization()
plot_error_ALS()
#make_table_ALS()

