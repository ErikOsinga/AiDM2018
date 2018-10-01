import numpy as np
import random
import sys
import assignment2 as a2

#def test_buckets():
#    max_bucket = 15
#    min_bucket = 5
#    num_buckets = max_bucket-min_bucket
#    size = np.logspace(3,7,base=10,num=100, dtype=np.int32)
#    results = np.empty([len(size)*num_buckets,3])
#    current_step=0
#    for num in range(len(size)):
#        print(num)
#        for buckets in range(min_bucket,max_bucket):
#            size_step = size[num%100]
#            results[current_step,0] = size_step
#            results[current_step,1] = buckets
#            results[current_step,2] = (a2.estimate_cardinality([random.randint(0, 2**32-1) for i in range(size_step)],buckets))
#            current_step += 1
#
#    np.savetxt("results.csv", results)
#
#def test_accuracy():
#    max_bucket = 15
#    min_bucket = 5
#    num_buckets = max_bucket-min_bucket
#    size = [10**3, 10**4, 10**5, 10**6]
#    results = np.empty([len(size)*num_buckets,4])
#    current_step=0
#    for num in range(len(size)):
#        for buckets in range(min_bucket,max_bucket):
#            temp = np.empty(10)
#            acc = np.empty(10)
#            size_step = size[num%4]
#            for j in range(10):
#                s = [random.randint(0, 2**32-1) for i in range(size_step)]
#                temp[j] = (a2.estimate_cardinality(s,buckets))
#                u = np.unique(s)
#                acc[j] = u/temp[j]
#            results[current_step,0] = size_step
#            results[current_step,1] = buckets
#            results[current_step,2] = np.mean(acc)
#            results[current_step,3] = np.std(acc)
#            current_step += 1
#            
#
#    np.savetxt("results_accuracy.csv", results)

def test_FM(num_iter,size_list):
    '''
    Test the FM function by running it num_iter times and taking the median of the outcome
    hash_groups = the number of groups the hashfunction is partitioned in (small multiple of log2(size))
    We generate a random number sequence with numbers of lengt 32bits and different sequence lengths proportional to the size_list
    The random sequence is probed for unique elements and the first "size" unique elements are fed into the cardinality estimator
    '''
    e = np.zeros(len(size_list)) # result for size in size_list
    for k in range(len(size_list)):
        print(k)
        size = size_list[k]
        e_t = [] # mean for every bucket
        hash_groups = int(2*np.log2(size))
        for j in range(num_iter): # run the algorithm num_iter times
            e_tt = [] # cardinality for every hash func
            for r in range(hash_groups): # use hash_groups different hash functions
                #print progress
                # sys.stdout.write('\r')
                # sys.stdout.write("[%-20s] %d%%" % ('='*int(r/hash_groups*20), r/hash_groups*100))
                # sys.stdout.flush()

                ss = a2.generate_R_distinct_values(size)
                e_tt.append(a2.estimate_cardinality_FM(ss))
            # // for r 
            # sys.stdout.write('\n')
            e_t.append(np.median(e_tt)) 
        # // for j
        e[k] = np.mean(e_t)    
    return e
 
size_list = [10**3, 10**4, 10**5, 10**6]

estimate = test_FM(num_iter=50,size_list=size_list)
print ('Estimates:', estimate)
print(np.abs((size_list-estimate)/size_list))

