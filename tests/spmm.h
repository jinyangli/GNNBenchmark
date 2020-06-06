#ifndef SPMM_H_
#define SPMM_H_

#include <cassert>
#include "./samples_io.h"
#include "./samples_utils.h"

namespace spmm {

struct GData {
  int D = 0;  // feat size
  float* ndata{nullptr};  // N*D
  float* weight{nullptr}; // M
  float* out{nullptr};    // N*D
};


struct SPMMFunctor {
  static inline bool CondEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    return true;
  }

  static inline void ApplyEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    for (int32_t fid = 0; fid < gdata->D; ++fid) {
#pragma omp atomic
      gdata->out[dst * gdata->D + fid] += gdata->ndata[src * gdata->D + fid] *
        gdata->weight[eid];
    }
  }
};

void InitGData(const utils::SampleCsr& csr, GData* gdata, GData* truth) {
  const int32_t N = csr.row_offsets.size() - 1;
  const int32_t M = csr.column_indices.size();
  const int D = gdata->D;
  std::vector<float> ndata(N * gdata->D), weight(M, 0.);
  for (size_t i = 0; i < ndata.size(); ++i) {
    ndata[i] = (float)rand() / RAND_MAX;
  }
  for (size_t i = 0; i < weight.size(); ++i) {
    // XXX: weight has to be the same across edges because transpose function did not change weights
    weight[i] = 3.45;
  }
  gdata->ndata = new float[N * D];
  gdata->weight = new float[M];
  gdata->out = new float[N * D];
  memcpy(gdata->ndata, &ndata[0], N * D * sizeof(float));
  memcpy(gdata->weight, &weight[0], M * sizeof(float));

  // compute truth
  truth->out = new float[N * D];
  std::fill(truth->out, truth->out + N * D, 0.);
  for (size_t u = 0; u < csr.row_offsets.size() - 1; u++) {
    for (int32_t eid = csr.row_offsets[u]; eid < csr.row_offsets[u + 1]; eid++) {
      int32_t v = csr.column_indices[eid];
      for (int32_t idx = 0; idx < D; idx++) {
        truth->out[v * D + idx] += ndata[u * D + idx] * weight[eid];
      }
    }
  }
}

void FreeGData(GData* gdata, GData* truth) {
  delete []truth->out;
  delete []gdata->out;
}

void CheckResult(const utils::SampleCsr& csr, GData* gdata, GData* truth) {
  const int32_t N = csr.row_offsets.size() - 1;
  const int D = gdata->D;
  bool equal = utils::IterEqual(gdata->out, truth->out, N * D);
  if (!equal) {
    for (int i = 0; i < N * D; ++i) {
      if (!FCLOSE(gdata->out[i], truth->out[i])) {
        std::cout << i << ": " << gdata->out[i] << " " << truth->out[i] << "\n";
      }    }
  }
  assert(equal);
  //std::cout << "Correct? " << equal << std::endl;
}

}  // masked_mm

#endif
