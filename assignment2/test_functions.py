def test_FM_2(size_list,amount_buckets):
    amount_buckets = int(amount_buckets)

    predictions = []
    for size in size_list:
        # make sure theres 2*np.log2 functions in every bucket
        num_hash = int(2*np.log2(size))*amount_buckets
        predictions.append(a2.estimate_cardinality_FM_combine_estimates(size, num_hash, amount_buckets))

    return np.asarray(predictions)

def estimate_cardinality_FM_combine_estimates(num_values, num_hash, split_number):
  """Estimates the number of unique elements in the input set values using the FM algortihm
  with grouping the hash functions into small groups, taking their average, and then taking
  the median of the averages.

  Arguments:
    num_values: The number of distinct elements to estimate cardinality of
    num_hash: The number of 'hash functions' that is used (number of random integer lists)
    split_number = the number of groups the hashfunction is partitioned in
  """

  # storing the trailing zeros for every hash function
  max_zeroes_hashfuncs = np.zeros(num_hash)
  
  for hashfunc in range(num_hash):
      # simulate a hash function by generating distinct values
      values = generate_R_distinct_values(num_values)
      max_zeroes = 0
      for h in values: # calculate for this hash function the max num trailing zeros
        max_zeroes = max(max_zeroes, trailing_zeroes(h))

      max_zeroes_hashfuncs[hashfunc] = max_zeroes 
    
  # then split the num_hash hash functions into split_number groups
  max_zeroes_split = np.asarray(np.split(max_zeroes_hashfuncs, split_number)) # list of splits
  predictions = 2**max_zeroes_split
  means = np.median(predictions,axis=0)
  prediction = np.mean(means)

  return prediction



