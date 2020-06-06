#include <iostream>
#include <cstdlib>
#include <time.h>
#include <sys/time.h>
#include <mkl_spblas.h>

#include <minigun/minigun.h>
#include "./spmm.h"

using minigun::advance::RuntimeConfig;
using namespace spmm;

double RunMinigun(const utils::SampleCsr& scsr,
                  const minigun::IntCsr& csr,
                  GData& gdata,
                  GData& truth) {
  // create stream
  RuntimeConfig rtcfg;
  rtcfg.ctx = {kDLCPU, 0};

  minigun::IntArray infront;

  // check accuracy
  typedef minigun::advance::Config<true, minigun::advance::kV2N> Config;
  minigun::advance::Advance<kDLCPU, int32_t, Config, GData, SPMMFunctor>(
      rtcfg, csr, &gdata, infront);
  CheckResult(scsr, &gdata, &truth);

  struct timespec start, end;

  // warm up
  const int K = 10;
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLCPU, int32_t, Config, GData, SPMMFunctor>(
        rtcfg, csr, &gdata, infront);
  }

  clock_gettime(CLOCK_REALTIME, &start);
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLCPU, int32_t, Config, GData, SPMMFunctor>(
        rtcfg, csr, &gdata, infront);
  }
  clock_gettime(CLOCK_REALTIME, &end);

  float dur = (end.tv_sec - start.tv_sec) * 1.0e3 + \
              (end.tv_nsec - start.tv_nsec) * 1.0e-6;
  return dur / K;
}

double RunMKL(utils::SampleCsr& scsr,
    GData& gdata,
    GData& truth) {
  // create stream

  const int32_t N = scsr.row_offsets.size() - 1;
  sparse_matrix_t A;
  sparse_status_t status;
  status = mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO, N, N,
      &scsr.row_offsets[0], &scsr.row_offsets[1], &scsr.column_indices[0],
      gdata.weight);
  assert(status == SPARSE_STATUS_SUCCESS);
  struct matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  status = mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr,
      SPARSE_LAYOUT_ROW_MAJOR, gdata.ndata, gdata.D, gdata.D, 0,
      gdata.out, gdata.D);
  assert(status == SPARSE_STATUS_SUCCESS);
  CheckResult(scsr, &gdata, &truth);

  struct timespec start, end;

  // warm up
  const int K = 10;
  for (int i = 0; i < K; ++i) {
    status = mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr,
        SPARSE_LAYOUT_ROW_MAJOR, gdata.ndata, gdata.D, gdata.D, 0,
        gdata.out, gdata.D);
    assert(status == SPARSE_STATUS_SUCCESS);
  }

  clock_gettime(CLOCK_REALTIME, &start);
  for (int i = 0; i < K; ++i) {
    status = mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr,
        SPARSE_LAYOUT_ROW_MAJOR, gdata.ndata, gdata.D, gdata.D, 0,
        gdata.out, gdata.D);
    assert(status == SPARSE_STATUS_SUCCESS);
  }
  clock_gettime(CLOCK_REALTIME, &end);

  float dur = (end.tv_sec - start.tv_sec) * 1.0e3 + \
              (end.tv_nsec - start.tv_nsec) * 1.0e-6;
  return dur / K;
}

int main(int argc, char** argv) {
  // test transpose
  srand(42);
  if (argc < 3) {
    std::cout << "USAGE: ./bench_spmm <file_name> <feat_size>" << std::endl;
    return 1;
  }
  const char* filename = argv[1];
  const int feat_size = std::atoi(argv[2]);

  utils::SampleCsr scsr;
  utils::LoadGraphFromFile(filename, &scsr);
  const int32_t N = scsr.row_offsets.size() - 1;
  const int32_t M = scsr.column_indices.size();

  // gdata
  GData gdata, truth;
  gdata.D = feat_size;
  InitGData(scsr, &gdata, &truth);

  // csr
  minigun::IntCsr csr = utils::ToMinigunCsr(scsr, kDLCPU);

  // csr
  double dur1 = RunMinigun(scsr, csr, gdata, truth);
  double dur2 = RunMKL(scsr, gdata, truth);
  std::cout << N << "," << M << "," << feat_size << "," << dur1 << ","
    << dur2 << "\n";

  FreeGData(&gdata, &truth);
  return 0;
}