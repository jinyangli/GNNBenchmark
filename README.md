Benchmark AMD CPU/GPU performance with GNN workload
===============================
This repository is for benchmarking sparse and dense kernel performance on AMD
CPU and GPU with Graph Neural Network (GNN) workload. For now, only CPU tests
are available here.


Setup the environment
---------------------

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

Build and run tests
-----------
### Setup this repository

First, `git clone` this repository and change directory into it. Then initialize submodules with 
```bash
git submodule update --init --recursive
```

### Build
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
|--------------|----------|----------|
| 16           | 1839.530 | 1324.340 |
| 32           | 2985.770 | 2380.760 |
| 64           | 4837.950 | 4560.380 |
| 128          | 9550.330 | 8952.170 |

For MKL SPMM kernel, the execution time in milliseconds:

| Feature Size | AMD      | Intel   |
|--------------|----------|---------|
| 16           | 277.550  | 114.241 |
| 32           | 552.329  | 101.318 |
| 64           | 1051.990 | 196.756 |
| 128          | 1958.280 | 670.561 |


Additional Tests
-----------------
### Dense matrix multiplication
[scripts/bench\_dense\_mm.py](./scripts/bench_dense_mm.py) benchmarks the
performance of multiplication between two dense matrix of size 1000 by 1000
using pytorch. To run this test, on Intel Machine, one needs to install torch.
And on AMD machine, use [this docker
file](https://github.com/ROCmSoftwarePlatform/pytorch/blob/master/docker/pytorch/cpu-only/Dockerfile)
from an AMD-maintained fork of PyTorch which uses BLIS as BLAS library.

We tested on p3.8xlarge (Intel CPU) and m5a.8xlarge (AMD CPU) instances on AWS.
For single precision matrix multiplication between two square matrices of size
1000x1000, on Intel CPU MKL takes 2.1-3.8ms, and AMD CPU BLIS takes about
4.7ms. We suspect that the large variance of Intel CPU is due to automatic CPU
clock rate adjustment.

Alternatively, one should use C++ interface of MKL and BLIS to compare their
performance.
