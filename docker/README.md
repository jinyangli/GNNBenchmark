Build docker image for benchmark
================================

Files in this folder
--------------------
Dockerfile for benchmarking sparse operations using GNN workload on AMD / Intel CPUs

install folder contains scripts that install common dependencies

Build image
---------------
```bash
docker build -t benchmark-sparse -f Dockerfile .
```

Run container
--------------
```bash
docker run -it --name benchmark benchmark-sparse
```
