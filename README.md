Benchmark AMD CPU/GPU performance with GNN workload
===============================
This repository is for benchmarking sparse and dense kernel performance on AMD
CPU and GPU with Graph Neural Network (GNN) workload. 

CPU Sparse Performance Benchmark
---------------------
Benchmark [DGL Minigun](https://github.com/dglai/minigun) sparse kernels and
MKL sparse kernels on AMD CPU and Intel CPU.

### Install system packages
Many packages need to be installed to build the tests and generate input for
the tests including build-essential, make, cmake, python3
and python packages like numpy, scipy, torch, dgl.

The list above is incomplete. We provide a DockerFile to build a container with
all required dependencies and we recommend you use it. Check out how to build
the container and run it in [docker](./docker).

### Install MKL
Download and install MKL for C/C++. This benchmark repository was tested with MKL\_2020.1.217

After installation, set up environment variable for MKL
```bash
export MKLROOT=/path/to/mkl
export CPATH=$CPATH:$MKLROOT/include
export LIBRARY_PATH=$LIBRARY_PATH:$MKLROOT/lib/intel64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MKLROOT/lib/intel64
```

When doing tested on AMD CPU, run the following command to improve MKL
performance as suggested in [this
post](https://www.pugetsystems.com/labs/hpc/How-To-Use-MKL-with-AMD-Ryzen-and-Threadripper-CPU-s-Effectively-for-Python-Numpy-And-Other-Applications-1637/)
```bash
export MKL_DEBUG_CPU_TYPE=5
```

### Build tests

First, `git clone` this repository and change directory into it. Then initialize submodules with 
```bash
git submodule update --init --recursive
```

Then, build the test by
```bash
cd /path/to/this/repository
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Generate input graph
```bash
cd /path/to/this/repository
cd scripts
```

The following commands use DGL package to download Reddit dataset and serialize the social graph for testing
```bash
mkdir bench-graphs
python3 gen_dgl_graph.py -o bench-graphs/reddit.grh --dataset=reddit-self-loop
```

If you want to try more graphs, simply follow `gen_dgl_graph.py` to serialize
your graphs. Also, DGL has [built-in
support](https://docs.dgl.ai/api/python/data.html#dataset-classes) for a large
set of datasets. Check out the list
[here](https://docs.dgl.ai/en/latest/features/dataset.html)

### Run tests
```bash
cd /path/to/this/repository
```
The executable we previously built is at `./build/tests/cpu_spmm`. It takes in
two arguments: input graph file and node feature size. The test code will
convert the input graph to a sparse matrix (A) in CSR format and creates a randomly
initialized node feature tensor (H) of size (num\_nodes, node\_feature\_size), and then
perform Sparse Matrix Multiplication (SPMM) between A and H and measure execution time.

If we are testing on AMD CPU, run the following:
```bash
export MKL_DEBUG_CPU_TYPE=5
```

Now, run the test:
```bash
./build/tests/cpu_spmm scripts/bench-graphs/reddit.grh 16
```

The testing code will check result correctness, warm up by executing the SPMM
10 times, and then test 10 times and report average execution time in
milliseconds.

### Results
The table below shows the results we got on Reddit Graph (232965 nodes,
114848857 edges) following the above steps using AWS machines. For Intel CPU we
used p3.8xlarge instance, and for AMD CPU we used m5a.8xlarge. Both have 32
virtual cores.

For Minigun SPMM kernel, the execution time in milliseconds:

| Feature Size | AMD      | Intel    |
|-------------:|---------:|---------:|
| 16           | 1839.530 | 1324.340 |
| 32           | 2985.770 | 2380.760 |
| 64           | 4837.950 | 4560.380 |
| 128          | 9550.330 | 8952.170 |

For MKL SPMM kernel, the execution time in milliseconds:

| Feature Size | AMD      | Intel   |
|-------------:|---------:|--------:|
| 16           | 277.550  | 114.241 |
| 32           | 552.329  | 101.318 |
| 64           | 1051.990 | 196.756 |
| 128          | 1958.280 | 670.561 |


GPU Sparse Performance Benchmark
-------------------------
Scripts in [tests-gpu](./tests-gpu) benchmarks performance of Sparse Matrix
Multiplication (SpMM) on AMD and NVIDIA GPU. The machines we used for benchmark
are: 

- AMD:
	- CPU: AMD EPYC 7452 32-Core Processor (128 virtual cores), 1.5GHz (max
	  2.35GHz), 1TB memory
	- GPU: Vega 20 [Radeon VII]: single precision 13.44 TFLOPS, 16GB
	  HBM2 memory, Bandwidth 1,024 GB/s, Memory Bus 4096 bit
- Intel / NVIDIA:
	- CPU: Intel(R) Core(TM) i7-9700 CPU (8 physical cores, 8 virtual
	  cores), 3.0GHz (max 4.6GHz), 32GB memroy
	- GPU: NVIDIA GeForce RTX 2080: single precision 10.07 TFLOPS, 8GB
	  GDDR6 memory,
  Bandwidth 448.0 GB/s, Memory Bus 256 bit

### Sparse Matrix Multiplication (SpMM) kernel

[tests-gpu/bench\_spmm.py](./tests-gpu/bench_spmm.py) benchmarks average
execution time (in milliseconds) of 100 runs (after warming up with another 100
runs). Below is the result using Reddit dataset as sparse graph:

| Feature Size | AMD    | Intel   |
|-------------:|-------:|--------:|
| 16           | 17.599 | 35.434  |
| 32           | 24.150 | 42.041  |
| 64           | 43.302 | 79.118  |
| 128          | 93.180 | 156.993 |

### End to end training time of Graph Convolution Network
[tests-gpu/gcn.py](./tests-gpu/gcn.py) benchmarks average epoch time (in
seconds) of training 2-layer GCN on Reddit Dataset with input feature size 602
and output feature size 41 and different hidden layer size. The accuracy
numbers show the mean and standard deviation of 10 runs.

| Hidden Size | AMD epoch time | AMD accuracy        | NIVIDA epoch time | NVIDIA accuracy     |
|------------:|---------------:|--------------------:|------------------:|--------------------:|
| 16          | 0.0692         | 78.99 &plusmn; 3.43 | 0.1811            | 78.46 &plusmn; 5.31 |
| 32          | 0.0762         | 90.21 &plusmn; 1.52 | 0.1886            | 88.42 &plusmn; 3.47 |
| 64          | 0.0982         | 92.62 &plusmn; 0.27 | 0.2270            | 92.51 &plusmn; 0.59 |
| 128         | 0.1520         | 93.24 &plusmn; 0.09 | 0.3078            | 93.24 &plusmn; 0.12 |


Additional Tests
-----------------
### Dense matrix multiplication
[scripts/bench\_dense\_mm.py](./scripts/bench_dense_mm.py) benchmarks the
performance of multiplication between two dense matrix of size 1000 by 1000
using PyTorch. To run this test, for Intel CPU or NVIDIA GPU, one needs to
install PyTorch. On AMD machines, for CPU, use [this docker
file](https://github.com/ROCmSoftwarePlatform/pytorch/blob/master/docker/pytorch/cpu-only/Dockerfile)
from an AMD-maintained fork of PyTorch which uses BLIS as BLAS library, and for
GPU, use [this recommended docker
image](https://rocmdocs.amd.com/en/latest/Deep_learning/Deep-learning.html#recommended-install-using-published-pytorch-rocm-docker-image).

We tested using the same machines mentioned in sparse kernel experiments above.
Average execution time (in milliseconds) of 10 runs is shown below. 

|     | AMD   | Intel / NVIDIA | 
|----:|------:|---------------:|
| CPU | 4.7   | 2.1-3.8        |
| GPU | 0.239 | 0.292          |

We suspect that the large variance of Intel CPU is due to automatic CPU clock
rate adjustment.

Alternatively, one should use C++ interface of MKL, BLIS, cuBLAS, and
rocBLAS to compare their performance.
