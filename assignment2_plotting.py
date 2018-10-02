import numpy as np
from matplotlib import pyplot as plt


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
   results = np.load('./results_assignment2.npy')
   num_groupss = np.load('./num_groupss.npy')
   small_ints = np.load('./small_ints.npy')
#   plt.imshow(results[0,:,:,0],origin='lower')
   plt.pcolor(results[0,:,:,0],cmap='Reds')
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
   plt.show()
   
loop_many_variables_plot_FM()  
