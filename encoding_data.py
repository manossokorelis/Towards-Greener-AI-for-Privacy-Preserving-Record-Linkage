# Importing necessary libraries
import pandas as pd
import numpy as np
from py_stringmatching import QgramTokenizer
from pybloom_live import BloomFilter
import random
from bitarray import bitarray
import pyRAPL
pyRAPL.setup()

# Declaring functions
def importingDatasets(file_pathA, file_pathB):
    """
    This function imports original and modified datasets from CSV files, 
    extracts the first column as the ID, selects only the next five columns, 
    and returns them as DataFrames.    

    Args:
    file_pathA (str): Path to the first CSV file.
    file_pathB (str): Path to the second CSV file.

    Returns:
    tuple: Two pandas DataFrames (dataA, dataB) with columns:
        - 'Id' (first column from the CSV)
        - 'Data' (concatenated string of the next five columns, separated by '|')
    """
    def importingDataset(file_path):
        data = pd.read_csv(file_path, header=None)
        id_df = data.iloc[:, 0].rename('Id')
        data_columns = data.iloc[:, 1:6].astype(str).agg('|'.join, axis=1)
        data_df = pd.DataFrame(data_columns, columns=['Data'])
        dataFrame = pd.concat([id_df, data_df], axis=1)
        return dataFrame
    dataA = importingDataset(file_pathA)
    dataB = importingDataset(file_pathB)
    return dataA, dataB

def generatingBloomFilters(dataA, dataB, flip_probability, capacity, error_rate, qval=2, padding=False):
    """
    This function generates Bloom filters for two datasets and optionally adds noise to them.
    
    Args:
    dataA (pd.DataFrame): First dataset containing an 'Id' column and a 'Data' column.
    dataB (pd.DataFrame): Second dataset containing an 'Id' column and a 'Data' column.
    flip_probability (float): Probability of flipping bits in the Bloom filter for added noise.
    capacity (int): Expected number of elements in the Bloom filter.
    error_rate (float): Desired false positive rate for the Bloom filter.
    qval (int): Value for q-gram tokenization.
    padding (bool, optional): Whether to use padding in q-gram tokenization. Defaults to False.
    
    Returns:
    tuple: Two pandas DataFrames containing 'Id' and 'BloomFilter' columns.
    """
    def randomized_response(bit, flip_probability):
        if random.random() < flip_probability:
           return 1 - bit
        return bit
    def add_noise_to_bloom_filter(bloom_filter, flip_probability):
        noisy_bits = [randomized_response(bit, flip_probability) for bit in bloom_filter]
        return bitarray(noisy_bits)
    def generatingBloomFilter(record, capacity, error_rate, padding, qval):
        bf = BloomFilter(capacity=capacity, error_rate=error_rate)
        qgrams = tokenizer.tokenize(str(record))
        for qgram in qgrams:
            bf.add(qgram)
        return bf.bitarray
    tokenizer = QgramTokenizer(padding=padding, qval=qval)
    if flip_probability > 0:
        dataA['BloomFilter'] = dataA['Data'].apply(lambda x:
            add_noise_to_bloom_filter(generatingBloomFilter(x, capacity, error_rate, padding, qval), flip_probability))
        dataB['BloomFilter'] = dataB['Data'].apply(lambda x:
            add_noise_to_bloom_filter(generatingBloomFilter(x, capacity, error_rate, padding, qval), flip_probability))
    else:
        dataA['BloomFilter'] = dataA['Data'].apply(lambda x: generatingBloomFilter(x, capacity, error_rate, padding, qval))
        dataB['BloomFilter'] = dataB['Data'].apply(lambda x: generatingBloomFilter(x, capacity, error_rate, padding, qval))
    return dataA[['Id', 'BloomFilter']], dataB[['Id', 'BloomFilter']]

def creatingPairs(bfA, bfB):
    """
    Creates all possible pairs of Bloom filters from two datasets and marks whether they match.

    Args:
    bfA (pd.DataFrame): First dataset containing 'Id' and 'BloomFilter'.
    bfB (pd.DataFrame): Second dataset containing 'Id' and 'BloomFilter'.

    Returns:
    pd.DataFrame: A DataFrame containing pairs of Bloom filters and a 'Matched' column indicating if they correspond to the same ID.
    """
    bfA.columns = ['IdA', 'BloomFilterA']
    bfB.columns = ['IdB', 'BloomFilterB']
    pairs = bfA.assign(key=1).merge(bfB.assign(key=1), on='key').drop('key', axis=1)
    pairs['Matched'] = (pairs['IdA'] == pairs['IdB']).astype(int)
    return pairs[['BloomFilterA', 'BloomFilterB', 'Matched']]

def creatingFeatures(pairs, batch_size):
    """
    Computes similarity and distance metrics for pairs of Bloom filters.

    Args:
    pairs (pd.DataFrame): DataFrame containing 'BloomFilterA', 'BloomFilterB', and 'Matched' columns.
    batch_size (int): Number of pairs to process in each batch to optimize memory usage.

    Returns:
    pd.DataFrame: A DataFrame containing Jaccard Similarity, Dice Similarity, and Hamming Distance for each pair.
    """
    def creatingFeaturesInBatches(pairs):
        bitarraysA = np.unpackbits(np.array(list(pairs['BloomFilterA'].values), dtype=np.uint8), axis=1)
        bitarraysB = np.unpackbits(np.array(list(pairs['BloomFilterB'].values), dtype=np.uint8), axis=1)
        common_bits = np.logical_and(bitarraysA, bitarraysB).sum(axis=1)
        union_bits = np.logical_or(bitarraysA, bitarraysB).sum(axis=1)
        sum_bits = bitarraysA.sum(axis=1) + bitarraysB.sum(axis=1)
        pairs['JaccardSimilarity'] = np.divide(common_bits, union_bits, out=np.zeros_like(common_bits, dtype=float), where=union_bits != 0)
        pairs['DiceSimilarity'] = np.divide(2 * common_bits, sum_bits, out=np.zeros_like(common_bits, dtype=float), where=sum_bits != 0)
        pairs['HammingDistance'] = (bitarraysA != bitarraysB).sum(axis=1)
        return pairs[['JaccardSimilarity', 'DiceSimilarity', 'HammingDistance', 'Matched']]
    results = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs.iloc[start:start + batch_size].copy()
        batch_results = creatingFeaturesInBatches(batch)
        results.append(batch_results)
    return pd.concat(results, ignore_index=True)

# Declaring parameters
faults_per_record = 1
capacity = 200
error_rate = 0.01
flip_probability = 0.01
##### OPTIMIZE MEMORY #####
batch_size = 1000000 
###########################

# Importing datasets
dataA, dataB = importingDatasets(file_pathA='/home/emmanouil-sokorelis/Thesis/datasets/POW_A_10000.csv', file_pathB=f'/home/emmanouil-sokorelis/Thesis/datasets/POW_B_{faults_per_record}_10000.csv')

# Generating differential privacy Bloom filters and measuring energy with PyRAPL
gbf = pyRAPL.Measurement('generatingBloomFilters')
gbf.begin()
bfA, bfB = generatingBloomFilters(dataA=dataA, dataB=dataB, flip_probability=flip_probability, capacity=capacity, error_rate=error_rate)
gbf.end()

# Creating all possible pairs
pairs = creatingPairs(bfA=bfA, bfB=bfB)

# Creating features (Jacaard similarity, Dice similarity, Hamming distance) and measuring energy with PyRAPL
cf = pyRAPL.Measurement('creatingFeatures')
cf.begin()
dataset = creatingFeatures(pairs=pairs, batch_size=batch_size)
cf.end()

# Results
print(f"Faults Per Record: {faults_per_record}, Capacity: {capacity}, Error Rate: {error_rate}, Flip probability: {flip_probability}")
print(f"Time for Generating Bloom Filter: {gbf.result.duration}")
print(f"CPU Energy Consumed on Generating Bloom Filter: {gbf.result.pkg}")
print(f"RAM Energy Consumed on Generating Bloom Filter: {gbf.result.dram}")
print(f"Time for Creating Features: {cf.result.duration}")
print(f"CPU Energy Consumed on Creating Features: {cf.result.pkg}")
print(f"RAM Energy Consumed on Creating Features: {cf.result.dram}")

# Storing dataset of similarity and difference scores
dataset.to_csv(f'/home/emmanouil-sokorelis/Thesis/datasets/encoding_pairs/dataset_fpr{faults_per_record}_f{flip_probability}_c{capacity}_er{error_rate}.csv', index=False, mode='w')
