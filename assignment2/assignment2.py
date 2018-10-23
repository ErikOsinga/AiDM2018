import numpy as np
import random
import time

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        #print('{:s} function took {:.3f} s'.format(f.__name__, (time2-time1)))

        return ret, time2-time1
    return wrap

def trailing_zeroes(num):
  """Counts the number of trailing 0 bits in num."""
  if num == 0:
    return 32 # Assumes 32 bit integer inputs!
  p = 0
  while (num >> p) & 1 == 0:
    p += 1
  return p

def generate_R_distinct_values(R):
    """
    Generates R distinct values between 0 and 2**32 - 1
    !! Make sure to not set a seed anywhere, else they will be the same every time
    which kind of destroys the purpose of having 'different' hash functions
    """
    distinct = np.random.randint(0,2**32-1,int(R*1.2))
    distinct = np.unique(distinct)
    distinct = distinct[:R]
    assert len(distinct) == R

    return distinct

#@timing
def estimate_cardinality_loglog(values, k):
  """Estimates the number of unique elements in the input set values using the LogLog algorithm

  Arguments:
    values: An iterator of hashable elements to estimate the cardinality of.
    k: The number of bits of hash to use as a bucket number; there will be 2**k buckets.
  """
  num_buckets = 2 ** k
  max_zeroes = [0] * num_buckets
  for h in values: 
    bucket = h & (num_buckets - 1) # Mask out the k least significant bits as bucket ID
    bucket_hash = h >> k
    max_zeroes[bucket] = max(max_zeroes[bucket], trailing_zeroes(bucket_hash))
  return 2 ** ( np.mean(max_zeroes)) * num_buckets * 0.79402

def estimate_cardinality_FM(values):
  """Estimates the number of unique elements in the input set values using the FM algortihm

  Arguments:
    values: An iterator of hashable elements to estimate the cardinality of.
  """
  max_zeroes = 0
  for value in values: 
    h = value
    max_zeroes = max(max_zeroes, trailing_zeroes(h))
  return 2 ** max_zeroes
































