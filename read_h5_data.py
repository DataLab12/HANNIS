import numpy as np
import os
import time
import h5py

def get_data_from_h5(h5_path):
    data_dct = {}
    with h5py.File(h5_path, 'r') as h5:
        # print('main video name: ', h5.attrs['video_name'])
        print('Loading h5 data from file...')


        data_group = h5.get('/feature')
        print('Dataaaaaaaaaa',data_group)
        keys = list(data_group.keys())
        j=0
        for k in keys:
            j += 1
            if j%1000==0:
                print('JJJJJJJJJ',j)
            # if j == 100000:
            #     break
            data = np.array(data_group.get(k))         # for 'data' feature
            # data = np.array(data_group.get(k))[0]       # for 'img_features' feature
            data_dct[k] = data
    return data_dct

def get_data(path):
    vectors = []
    frames = []
    t = time.time()
    data = get_data_from_h5(path)
    print('Data loading time: t = ', time.time()-t)

    frames = list(data.keys())
    for k in frames:
        v = data[k]
        vectors.append(v)

    #print(frames[:100])
    frames = np.array(frames)
    vectors = np.array(vectors).astype(np.float32)
    # print(vectors[0].shape)
    dim = vectors[0].shape[0]        
    print('Dimension is',dim)
    if not os.path.exists('indexes/'):
        os.system('mkdir indexes')

    return data,vectors,dim,frames