import numpy as np
from matplotlib import pyplot as plt


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
	ax.legend(fontsize=18, framealpha=0)
	plt.tight_layout()
	ax.tick_params(axis="both", which="both", bottom=False, top=False,    
                labelbottom=True, left=False, right=False, labelleft=True) 


''' 
def bucket_plots():
    ar = np.loadtxt("results.csv")
    fig, axes  = plt.subplots(nrows=2,ncols=5)
    max_bucket = 15
    min_bucket = 5
    for buckets in range(min_bucket,max_bucket):
        num_buck=buckets-min_bucket
        if num_buck<5:
            axes[0,num_buck].semilogx(ar[ar[:,1] == buckets,0], ar[ar[:,1] == buckets,0]/ar[ar[:,1] == buckets,2])

            axes[0,num_buck].set_title("bucket size {}".format(buckets))
        else:
            axes[1,num_buck-5].semilogx(ar[ar[:,1] == buckets,0], ar[ar[:,1] == buckets,0]/ar[ar[:,1] == buckets,2])

            axes[1,num_buck-5].set_title("bucket size {}".format(buckets))
    plt.show()

def accuracy_plots():
    ar = np.loadtxt("results_accuracy.csv")
    fig, axes  = plt.subplots(nrows=1,ncols=4)
    accuracy = np.abs(ar[:,2] - 1)
    size = [10**3, 10**4, 10**5, 10**6]
    for i in range(4):
        axes[i].scatter(ar[ar[:,0]==size[i],1],np.abs(ar[ar[:,0]==size[i],2]-1)*100)
        axes[i].errorbar(ar[ar[:,0]==size[i],1],np.abs(ar[ar[:,0]==size[i],2]-1)*100, yerr=ar[ar[:,0]==size[i],3], linestyle='None')
        axes[i].set_title("{}".format(size[i]))
    plt.show()
accuracy_plots()
''' 

def loop_many_variables_plot_FM():
   results = np.load('./results_assignment2_test6.npy')
   num_groupss = np.load('./num_groupss_test6.npy')
   small_ints = np.load('./small_ints_test6.npy')
   results_2 = np.load('./results_assignment2_test5.npy')
   num_groupss_2 = np.load('./num_groupss_test5.npy')
   small_ints_2 = np.load('./small_ints_test5.npy')
#   plt.imshow(results[0,:,:,0],origin='lower')

   plt.pcolormesh(results[0,:,:,0] ,cmap='Reds')
   # set x axis correctly showing num groups
   ticks = np.arange(0.5,len(num_groupss),1)
   labels = num_groupss
   plt.xticks(ticks, labels)
   plt.xlabel('Number of groups')
   
   # set y axis correctly showing small_ints
   ticks = np.arange(0.5,len(small_ints),1)
   labels = small_ints
   plt.yticks(ticks, labels)
   plt.ylabel('Small int')
   
   plt.colorbar()
   plt.savefig('FM_2D_plot_size_10^3_test6')
   plt.show()
   
def loop_many_variables_plot_loglog():
    results = np.load('./results_assignment2_loglog.npy')
    buckets = np.load('./buckets.npy')
    fig, ax = plt.subplots()
    for i in range(len(results[:,0][:,0])):
        ax.errorbar(range(len(results[i,:][:,0])), results[i,:][:,0], yerr = results[i,:][:,1], fmt='o', capsize=5, label='$\mathrm{10^%s}$'%(i+3))
    for y in np.geomspace(0.001, 10**5, num=9):    
        ax.plot(range(len(buckets)), [y] * len(range(len(buckets))), "--", lw=0.5, color="black", alpha=0.3)   	

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_params(ax,r'$\mathrm{\log_2}$( Number of buckets )','Relative error(RAE)','')
    plt.xticks(range(0, len(results[i,:][:,0]), 2), range(np.min(buckets),np.max(buckets),2))
    plt.yscale('log')
    plt.savefig('logscale_buckets')
    plt.show()
   
#loop_many_variables_plot_loglog()  
loop_many_variables_plot_FM()