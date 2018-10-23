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

@a2.timing
def test_FM(num_groups,size,small_int):
    '''
    Test the FM function by running it num_iter times and taking the median of the outcome
    hash_groups = the number of groups the hashfunction is partitioned in (small multiple of log2(size))
    small_int = the small multiple that decides the number of hash groups 
    We generate a random number sequence with numbers of lengt 32bits and different sequence lengths proportional to the size_list
    The random sequence is probed for unique elements and the first "size" unique elements are fed into the cardinality estimator
    '''
    e_t = [] # mean for every bucket
    hash_groups = int(small_int*np.log2(size))
    # for j in range(num_groups): # run the algorithm num_iter times (create groups)
        # e_tt = [] # cardinality for every hash func
        # for r in range(hash_groups): # use hash_groups different hash functions
    
    # APPARENTLY THIS WORKS BETTER ?
    for j in range(hash_groups): # run the algorithm num_iter times (create groups)
        e_tt = [] # cardinality for every hash func
        for r in range(num_groups): # use hash_groups different hash functions
           
           
            ss = a2.generate_R_distinct_values(size)
            e_tt.append(a2.estimate_cardinality_FM(ss))
            
        # // for r 
        e_t.append(np.median(e_tt)) 
    # // for j
    e = np.mean(e_t)    
    RAE = abs((size-e)/size)
    return RAE
    
def loop_many_variables_FM(size_list):
    small_ints = [2,4,8]
    num_groupss = [5,15,25,35]
    # (size_list,small_int,num_groupss) : tuple (RAE,time)
    results = np.empty((len(size_list),len(small_ints),len(num_groupss)),dtype=(float,2))  
    
    for k, size in enumerate(size_list):
        print('Now doing size:',size)    
        for i, small_int in enumerate(small_ints):
            for j, num_groups in enumerate(num_groupss):
                # because the results vary very much for random hashes, 
                # we choose to do 10 iterations of the same, and take the median
                tmp_RAE = []
                tmp_time = []
                for FMiter in range(10):
                    RAE, time = test_FM(num_groups, size, small_int)
                    tmp_RAE.append(RAE)
                    tmp_time.append(time)
                # save the result for this size,small_int,num_group combination
                results[k,i,j] = (np.median(tmp_RAE),np.mean(tmp_time))
                    
    np.save('./results_assignment2_test7.npy',results)
    np.save('./small_ints_test7.npy',small_ints)
    np.save('./num_groupss_test7.npy',num_groupss)
    np.save('./size_list_test7.npy',size_list)
 


def loop_many_variables_loglog(size_list, buckets, num_iter):
    results = np.empty((len(size_list),len(buckets),),dtype=(float,3))  
    
    for k,size in enumerate(size_list):
        print('Now doing size: {}'.format(size))
        for i, bucketss in enumerate(buckets):
            tmp_RAE = []
            tmp_time = []
            for j in range(num_iter):
                ss = a2.generate_R_distinct_values(size)
                e, time = a2.estimate_cardinality_loglog(ss, bucketss)
                tmp_RAE.append(abs((size-e)/size))
                tmp_time.append(time)
            results[k,i] = (np.mean(tmp_RAE), np.std(tmp_RAE), np.mean(tmp_time))
    np.save('./results_assignment2_loglog.npy', results)
    np.save('./buckets.npy', buckets)
    

if __name__ == '__main__':
    size_list = [10**3]#, 10**4, 10**5]#, 10**6]

    # loop_many_variables_loglog(size_list, [2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23], 20)
    loop_many_variables_FM(size_list)               


