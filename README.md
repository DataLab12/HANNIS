# Indexing and Searching in high-dimensional deep descriptor databases

hannis.py is a script for indexing and searching in high-dimensional deep descriptor databases that are saved in a hdf5 data format. Here,the results are evaluated based on Precision and Searching time. Precision is measured against sklearn.neighbors' brute search method.

## Installation
* numpy
* sklearn
* h5py
* ml_metrics
* time

## Cython file build
Run **setup.py** wth command  **python setup.py build_ext --inplace**
Move the **kmeanspp.pyx** and **hnsw.pyx** files to **dep** folder

## Indexing and Searching

Step1:  Load the h5/fbin/fvecs data in data_path.  
Step2:  Input a test frame number (e.g. 100) for h5 or an index number for fbin/fvecs to search  
Step3:  Enter the number of clusters to build
Step4:  Input the number of nearest neighbors to return (e.g. 100)  
Step5:  A folder named 'indexes' will be created for the first time and indexes will be saved inside the folder.  
Step6:  Enter the number of clusters to load from the indexes.
Step7:  Repeat Step2 to Step6 to get nearest neighbors for different frames/features.
