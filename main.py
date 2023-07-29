## import libraries
import numpy as np

from sklearn.datasets import make_blobs
import read_fbin
import combine_kmeanspp_and_hnsw
import os
import time

index_path = 'indexes/'
index_name = 'hannis'
if not os.path.exists(index_path):
    os.system('mkdir indexes')

data_path = './DOTAsmall.fbin'

# Generate dataset
# vectors, Y = make_blobs(n_samples=500, n_features=30)

vectors,dim = read_fbin.read_fbin(data_path)

print(f'Data shape: {vectors.shape}')


ncls =  5           # The number of classes (clusters)

init = 'k-means++'      # 'k-means++', 'random'
max_iter = 10           # number of iterations for clustering
m=16                    # number of neighboring nodes
ef=200                  # depth of search on a node

k = 10                  # number of nearest neighbors to return
clus_to_load = 1        # number of clusters to search

query = vectors[123]    # sample query vector

centroid_path = index_path + index_name
if not os.path.exists(centroid_path):
    combine_kmeanspp_and_hnsw.save_indexes_and_centers(index_name,index_path,vectors,ncls,init,max_iter,m,ef)

t=time.time()
predicted = combine_kmeanspp_and_hnsw.search_indexes(index_name,index_path,query,k,clus_to_load)
print('Total search time: ',time.time()-t)

print('Predicted: ',predicted)

