import numpy as np
import random
import assignment2 as a2


num_buckets = 12-7
size = np.logspace(3,7,base=10,num=50, dtype=np.int32)
results = np.empty([3, len(size)*num_buckets])
for num in range(len(size)*num_buckets):
    print(num)
    for buckets in range(7,12):
        size_step = size[num]
        results[0,num] = buckets
        results[1,num] = size_step
        results[2,num] = (a2.estimate_cardinality([random.randint(0, 2**32-1) for i in range(size_step)],10))

np.savetxt("results.csv", results)
