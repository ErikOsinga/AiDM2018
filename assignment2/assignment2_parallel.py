import numpy as np
import random
import sys
import assignment2 as a2

from multiprocessing import Pool
import itertools

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

    for j in range(num_groups): # run the algorithm num_iter times (create groups)
        e_tt = [] # cardinality for every hash func
        for r in range(hash_groups): # use hash_groups different hash functions
    
    # APPARENTLY THIS WORKS BETTER not true
    # for j in range(hash_groups): # run the algorithm num_iter times (create groups)
    #     e_tt = [] # cardinality for every hash func
    #     for r in range(num_groups): # use hash_groups different hash functions
           
           
            ss = a2.generate_R_distinct_values(size)
            e_tt.append(a2.estimate_cardinality_FM(ss))
            
        # // for r 
        e_t.append(np.median(e_tt)) 
    # // for j
    e = np.mean(e_t)    
    RAE = abs((size-e)/size)
    return RAE
    
def loop_many_variables_FM(params):
    # (size_list,small_int,num_groupss) : tuple (RAE,time)
    
    # result,small_int,num_groups,size
    results = []
    print params
    size,small_int,num_groups = params[0], params[1], params[2]

    # because the results vary very much for random hashes, 
    # we choose to do 10 iterations of the same, and take the median
    tmp_RAE = []
    for FMiter in range(10):
        RAE = test_FM(num_groups, size, small_int)
        tmp_RAE.append(RAE)
        # save the result for this size,small_int,num_group combination
        results.append(np.median(tmp_RAE))
                
    results.append(small_int)
    results.append(num_groups)
    results.append(size)

    return results

if __name__ == '__main__':
    size_list = itertools.chain(
        range(10**3,10**3+1),range(10**4,10**4+1),range(10**5,10**5+1),
        range(10**6,10**6+1))
    small_ints = range(2,8,1)
    num_groupss = range(2,16,1)

    paramlist = list(itertools.product(size_list,small_ints,num_groupss))


    pool = Pool(multiprocessing.cpu_count()-1)
    res = pool.map(loop_many_variables_FM, paramlist)    

    np.save('./result_multiprocessing',res)


