# SG
Stratified Graph for Indexing and Search of Large Deep Descriptor Databases

A python library for high-dimensional indexing and searching

### Python scripts description 
* [hlg.py](https://github.com/DataLab12/HANNIS/blob/main/hlg.py) contains the algorithm.
* [main.py](https://github.com/DataLab12/HANNIS/blob/main/main.py) is an example of indexing and searching.
* The [read_DEEP.py](https://github.com/DataLab12/HANNIS/blob/main/read_DEEP.py), [read_SIFT.py](https://github.com/DataLab12/HANNIS/blob/main/read_SIFT.py) and [read_glove.py](https://github.com/DataLab12/HANNIS/blob/main/read_glove.py) are the scripts for reading the *.fbin and *.fvecs and glove data.
* [evaluation.py](https://github.com/DataLab12/HANNIS/blob/main/evaluation.py) contains the evaluation metrics.
* [DOTAsmall.fbin](https://github.com/DataLab12/HANNIS/blob/main/DOTAsmall.fbin) is provided as a sample deep descriptor dataset extracted from DOTA2.0 aerial image dataset using ResNet50.  

### Algorithm parameters
* `ef`- the size of the dynamic list for the nearest neighbors (used during the search). Higher ef leads to more accurate but slower search. `ef` cannot be set lower than the number of queried nearest neighbors `k`. The value `ef` of can be anything between `k` and the size of the dataset.
* `k`- the number of nearest neighbors to be returned as the result.
* `m`- the number of bi-directional links created for every new element during construction. Higher `m` works better for high-dimensional dataset. It also determines the index size and performance efficiency.

### How to use the library

* Direct to the folder **HANNIS** and run the `main.py`

```
cd path/of/HANNIS
python main.py
```
* Input the number of number of nearest neighbor to retrieve.

### Installation
* numpy
* sklearn
* pickle
* scipy
* heapq 
