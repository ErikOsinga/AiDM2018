import numpy as np
import random
import assignment2 as a2

def test_buckets():
    max_bucket = 15
    min_bucket = 5
    num_buckets = max_bucket-min_bucket
    size = np.logspace(3,7,base=10,num=100, dtype=np.int32)
    results = np.empty([len(size)*num_buckets,3])
    current_step=0
    for num in range(len(size)):
        print(num)
        for buckets in range(min_bucket,max_bucket):
            size_step = size[num%100]
            results[current_step,0] = size_step
            results[current_step,1] = buckets
            results[current_step,2] = (a2.estimate_cardinality([random.randint(0, 2**32-1) for i in range(size_step)],buckets))
            current_step += 1

    np.savetxt("results.csv", results)

def test_accuracy():
    max_bucket = 15
    min_bucket = 5
    num_buckets = max_bucket-min_bucket
    size = [10**3, 10**4, 10**5, 10**6]
    results = np.empty([len(size)*num_buckets,4])
    current_step=0
    for num in range(len(size)):
        for buckets in range(min_bucket,max_bucket):
            temp = np.empty(10)
            acc = np.empty(10)
            size_step = size[num%4]
            for j in range(10):
                s = [random.randint(0, 2**32-1) for i in range(size_step)]
                temp[j] = (a2.estimate_cardinality(s,buckets))
                u = np.unique(s)
                acc[j] = u/temp[j]
            results[current_step,0] = size_step
            results[current_step,1] = buckets
            results[current_step,2] = np.mean(acc)
            results[current_step,3] = np.std(acc)
            current_step += 1
            

    np.savetxt("results_accuracy.csv", results)

test_accuracy()
