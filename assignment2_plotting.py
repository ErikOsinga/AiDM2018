import numpy as np
from matplotlib import pyplot as plt



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

