import kmeanspp
import hnsw
import numpy as np
import collections
import pickle
import time
import os
import heapq


def l2_distance( a, b):
    return np.linalg.norm(a - b)

def vectorized_distance( x, ys):
        return [l2_distance(x, y) for y in ys]

def get_indexes_for_clusters(vectors,ncls,init,max_iter):
    
    clusters, centers,distances = kmeanspp.func_Kmeans(vectors, ncls, init, max_iter)
    # print(f'Clusters: {clusters}')
    dict_of_clus = collections.defaultdict(list)
    dict_of_dist = collections.defaultdict(list)
    idx = 0
    for clus,dist in zip(clusters,distances):
        dict_of_clus[clus].append(idx)
        dict_of_dist[clus].append(dist)
        idx+=1

    return dict_of_clus,dict_of_dist,centers

def save_indexes_and_centers(name,index_path,vectors,ncls,init,max_iter,m,ef_construction):
    t = time.time()
    clus_dict,dist_dict,centers = get_indexes_for_clusters(vectors,ncls,init,max_iter)
    # print('Centers: ',type(centers),centers[0])
    print('Clustering time: ', time.time()-t,'seconds')
    centers_path = f"{index_path}{name}"
    if not os.path.exists(centers_path):
        with open(centers_path, 'wb') as f:
            pickle.dump(centers,f,pickle.HIGHEST_PROTOCOL)

    # building hnsw indexes for each clusters

    print('Indexing...')
    tot = time.time()
    for n in range (ncls):
        idx_path = f"{index_path}{name}_{n}"
        if not os.path.exists(idx_path):
            centroid = centers[n]
            curr_indxes = clus_dict[n]
            curr_dist = dist_dict[n]
            vec_to_cen_dist = []
            for d in curr_dist:
                vec_to_cen_dist.append(d[n])

            hnsw_index = hnsw.HNSW('l2',m,ef_construction)
            t =time.time()
            for  i in curr_indxes:
                hnsw_index.add(i,vectors[i])
            print('Hannis index saving time for cluster: ',n,' is: ', time.time()-t,'seconds')
        with open(idx_path, 'wb') as f:
            picklestring = pickle.dump(hnsw_index, f, pickle.HIGHEST_PROTOCOL)
    
          
    print('Total indexing time with hannis: ',' is: ', time.time()-tot,'seconds')

            

        
def search_indexes(name,index_path,query,n_neighbors,clus_to_load):
    
    centers_path = f"{index_path}{name}"
    t =time.time()
    fl = open(centers_path,'rb')
    centers = pickle.load(fl)
    print('Center loading time is: ', time.time()-t,'seconds')
    distances = collections.defaultdict(list)

    t= time.time()
    for index,c in enumerate(centers):
        dist = l2_distance(c,query)
        distances[index].append(dist)

    sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1]))
    print('Distance sorting time is: ', time.time()-t,'seconds')

    print('Sorted distances: ',sorted_distances)
    final_data = []
    for i in range (clus_to_load):
        load_index=list(sorted_distances.keys())[i]
        print('Load index: ', load_index)
        idx_path = f"{index_path}{name}_{load_index}"
        fl = open(idx_path,'rb')
        t=time.time()
        index = pickle.load(fl)
        print('Index loading time is: ', time.time()-t,'seconds')
        t=time.time()
        idx = index.search(query,k=n_neighbors)
        final_data.extend(idx)
    
    sorted_final_data = sorted(final_data, key=lambda x: x[1])

 
    return sorted_final_data[:n_neighbors]