import numpy as np

def reject_outliers(data,f):

    u = np.mean(data,axis=0)
    distances = []
    distances = np.linalg.norm(data - u[:], axis=1)
    std = np.std(distances)
    max_dist = max(distances)
    min_dist = min(distances)
    avg_dist = sum(distances)/len(distances)
    lower_bound = min_dist
    upper_bound = avg_dist + f*std
    # print('Max: ',max_dist,' Min: ',min_dist, ' Avg:',avg_dist, ' Std all: ',std,' Exp:',avg_dist+f*std)
    
    return lower_bound,upper_bound,distances
