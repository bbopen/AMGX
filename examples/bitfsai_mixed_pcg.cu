// SPDX-FileCopyrightText: 2026 Brett Bonner
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Standalone example & verification harness for AMGX BITFSAI smoother:
//   - Exercises native AMGX JSON parameters (bitfsai_mode, bitfsai_omega, bitfsai_safe_omega)
//   - Demonstrates FP32 V-cycle (AMGX_mode_dFFI) inside FP64 outer PCG with runtime
//     breakdown/stagnation detection (r^T z <= 0, curvature, drift) and automatic
//     recovery via FP64 V-cycle (AMGX_mode_dDDI)
//   - Includes `--self-test` verifying injected negative r^T z recovery and config modes

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <amgx_c.h>

#define CHECK_CUDA(expr)                                                               \
  do {                                                                                 \
    cudaError_t err = (expr);                                                          \
    if (err != cudaSuccess)                                                            \
      throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(err));   \
  } while (0)

#define CHECK_AMGX(expr)                                                               \
  do {                                                                                 \
    AMGX_RC rc = (expr);                                                               \
    if (rc != AMGX_RC_OK)                                                              \
      throw std::runtime_error(std::string(#expr) + " failed rc=" + std::to_string(rc)); \
  } while (0)

__global__ void spmv_f64_kernel(int n, const int *rp, const int *ci, const double *val,
                                const double *x, double *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double s = 0.0;
  for (int k = rp[i]; k < rp[i + 1]; ++k) s += val[k] * x[ci[k]];
  y[i] = s;
}

__global__ void axpy_f64_kernel(int n, double a, const double *x, double *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += a * x[i];
}

__global__ void xpay_f64_kernel(int n, const double *z, double beta, double *p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = z[i] + beta * p[i];
}

__global__ void cast_d2f_kernel(int n, const double *src, float *dst) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = (float)src[i];
}

__global__ void cast_f2d_kernel(int n, const float *src, double *dst) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = (double)src[i];
}

__global__ void negate_f64_kernel(int n, double *v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) v[i] = -v[i];
}

__global__ void dot_f64_kernel(int n, const double *x, const double *y, double *out) {
  __shared__ double sh[256];
  double s = 0.0;
  for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256)
    s += x[i] * (y ? y[i] : 1.0);
  int t = threadIdx.x;
  sh[t] = s;
  __syncthreads();
  for (int st = 128; st > 0; st >>= 1) {
    if (t < st) sh[t] += sh[t + st];
    __syncthreads();
  }
  if (t == 0) out[blockIdx.x] = sh[0];
}

struct DevicePCGBuffers {
  int n = 0;
  double *d_r = nullptr, *d_z = nullptr, *d_p = nullptr, *d_q = nullptr;
  double *d_part = nullptr, *d_scal = nullptr;
  float *d_rf = nullptr, *d_zf = nullptr;

  void alloc(int n_) {
    n = n_;
    CHECK_CUDA(cudaMalloc(&d_r, (size_t)n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_z, (size_t)n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_p, (size_t)n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_q, (size_t)n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_part, 512 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_scal, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_rf, (size_t)n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_zf, (size_t)n * sizeof(float)));
  }

  ~DevicePCGBuffers() {
    cudaFree(d_r); cudaFree(d_z); cudaFree(d_p); cudaFree(d_q);
    cudaFree(d_part); cudaFree(d_scal); cudaFree(d_rf); cudaFree(d_zf);
  }

  double dot(const double *x, const double *y) {
    int bl = std::min(512, (n + 255) / 256);
    dot_f64_kernel<<<bl, 256>>>(n, x, y, d_part);
    dot_f64_kernel<<<1, 256>>>(bl, d_part, nullptr, d_scal);
    double v = 0.0;
    CHECK_CUDA(cudaMemcpy(&v, d_scal, sizeof(double), cudaMemcpyDeviceToHost));
    return v;
  }
};

struct SolveOutcome {
  int iters = 0;
  bool recovered = false;
  std::string trigger = "none";
  double true_rel = INFINITY;
  float solve_ms = 0.0f;
};

static std::string make_vcycle_json(const std::string &smoother, const std::string &mode,
                                    double omega, int safe_omega) {
  std::string sm = (smoother == "BITFSAI")
                       ? ("\"smoother\":{\"solver\":\"BITFSAI\",\"bitfsai_mode\":\"" + mode +
                          "\",\"bitfsai_omega\":" + std::to_string(omega) +
                          ",\"bitfsai_safe_omega\":" + std::to_string(safe_omega) + "}")
                       : "\"smoother\":\"JACOBI_L1\"";
  return "{\"config_version\":2,\"determinism_flag\":0,\"solver\":{"
         "\"solver\":\"AMG\",\"algorithm\":\"CLASSICAL\",\"selector\":\"PMIS\","
         "\"interpolator\":\"D2\",\"cycle\":\"V\",\"presweeps\":1,\"postsweeps\":1,"
         "\"coarsest_sweeps\":2,\"max_iters\":1,\"monitor_residual\":0,"
         "\"store_res_history\":0," + sm + "}}";
}

// Mixed-precision FP64 outer PCG wrapping an FP32 AMG V-cycle (AMGX_mode_dFFI)
// with runtime breakdown detection and automatic FP64 V-cycle (AMGX_mode_dDDI) recovery.
static SolveOutcome solve_mixed_pcg_with_fallback(
    int n, const int *d_rp, const int *d_ci, const double *d_val,
    const double *d_b, double *d_x,
    AMGX_solver_handle solver_f32, AMGX_vector_handle vf_b, AMGX_vector_handle vf_x,
    AMGX_solver_handle solver_f64, AMGX_vector_handle vd_b, AMGX_vector_handle vd_x,
    DevicePCGBuffers &buf, double tol, int max_iters, bool inject_neg_rz = false) {
  SolveOutcome out;
  int bl = (n + 255) / 256;
  cudaEvent_t ev0, ev1;
  CHECK_CUDA(cudaEventCreate(&ev0));
  CHECK_CUDA(cudaEventCreate(&ev1));
  CHECK_CUDA(cudaEventRecord(ev0));

  CHECK_CUDA(cudaMemset(d_x, 0, (size_t)n * sizeof(double)));
  CHECK_CUDA(cudaMemcpy(buf.d_r, d_b, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));
  double b2 = buf.dot(d_b, d_b);
  double bnorm = std::sqrt(b2);

  for (int stage = 0; stage < 2; ++stage) {
    if (stage == 1) {
      out.recovered = true;
      spmv_f64_kernel<<<bl, 256>>>(n, d_rp, d_ci, d_val, d_x, buf.d_q);
      CHECK_CUDA(cudaMemcpy(buf.d_r, d_b, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));
      axpy_f64_kernel<<<bl, 256>>>(n, -1.0, buf.d_q, buf.d_r);
      double r2 = buf.dot(buf.d_r, buf.d_r);
      if (!std::isfinite(r2)) {
        CHECK_CUDA(cudaMemset(d_x, 0, (size_t)n * sizeof(double)));
        CHECK_CUDA(cudaMemcpy(buf.d_r, d_b, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));
      }
    }

    double prev_rho = 0.0, best_rel = INFINITY;
    int stale = 0;
    std::string reason = "max_iters";
    for (int it = 0; it < max_iters; ++it) {
      if (stage == 0) {
        cast_d2f_kernel<<<bl, 256>>>(n, buf.d_r, buf.d_rf);
        CHECK_AMGX(AMGX_vector_upload(vf_b, n, 1, buf.d_rf));
        CHECK_AMGX(AMGX_vector_set_zero(vf_x, n, 1));
        CHECK_AMGX(AMGX_solver_solve(solver_f32, vf_b, vf_x));
        CHECK_AMGX(AMGX_vector_download(vf_x, buf.d_zf));
        cast_f2d_kernel<<<bl, 256>>>(n, buf.d_zf, buf.d_z);
        if (inject_neg_rz && it == 0) negate_f64_kernel<<<bl, 256>>>(n, buf.d_z);
      } else {
        CHECK_AMGX(AMGX_vector_upload(vd_b, n, 1, buf.d_r));
        CHECK_AMGX(AMGX_vector_set_zero(vd_x, n, 1));
        CHECK_AMGX(AMGX_solver_solve(solver_f64, vd_b, vd_x));
        CHECK_AMGX(AMGX_vector_download(vd_x, buf.d_z));
      }

      double rho = buf.dot(buf.d_r, buf.d_z);
      if (!(rho > 0.0 && std::isfinite(rho))) { reason = "rz"; break; }

      double beta = (it == 0) ? 0.0 : (rho / prev_rho);
      if (it == 0) CHECK_CUDA(cudaMemcpy(buf.d_p, buf.d_z, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));
      else xpay_f64_kernel<<<bl, 256>>>(n, buf.d_z, beta, buf.d_p);

      spmv_f64_kernel<<<bl, 256>>>(n, d_rp, d_ci, d_val, buf.d_p, buf.d_q);
      double den = buf.dot(buf.d_p, buf.d_q);
      if (!(den > 0.0 && std::isfinite(den))) { reason = "curvature"; break; }

      double alpha = rho / den;
      axpy_f64_kernel<<<bl, 256>>>(n, alpha, buf.d_p, d_x);
      axpy_f64_kernel<<<bl, 256>>>(n, -alpha, buf.d_q, buf.d_r);
      out.iters++;

      double rel = std::sqrt(buf.dot(buf.d_r, buf.d_r)) / bnorm;
      prev_rho = rho;
      if (rel <= tol * 0.8) {
        spmv_f64_kernel<<<bl, 256>>>(n, d_rp, d_ci, d_val, d_x, buf.d_q);
        CHECK_CUDA(cudaMemcpy(buf.d_r, d_b, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));
        axpy_f64_kernel<<<bl, 256>>>(n, -1.0, buf.d_q, buf.d_r);
        double true_rel = std::sqrt(buf.dot(buf.d_r, buf.d_r)) / bnorm;
        if (std::isfinite(true_rel) && true_rel <= tol) {
          out.true_rel = true_rel;
          reason = "ok";
          break;
        }
        reason = "residual_drift";
        break;
      }
      if (rel < best_rel * 0.99) { best_rel = rel; stale = 0; }
      else if (++stale >= 100) { reason = "stagnation"; break; }
    }
    if (reason == "ok") break;
    if (stage == 0) out.trigger = reason;
  }

  CHECK_CUDA(cudaEventRecord(ev1));
  CHECK_CUDA(cudaEventSynchronize(ev1));
  CHECK_CUDA(cudaEventElapsedTime(&out.solve_ms, ev0, ev1));
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  return out;
}

static void run_self_test() {
  // Build small 16^3 3D anisotropic SPD Poisson system
  const int N = 16, n = N * N * N;
  std::vector<int> rp(n + 1, 0), ci;
  std::vector<double> val;
  std::vector<float> valf;
  auto idx = [&](int x, int y, int z) { return (z * N + y) * N + x; };
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        int r = idx(x, y, z);
        double diag = 1e-4;
        std::vector<std::pair<int, double>> row;
        auto add = [&](int nx, int ny, int nz, double w) {
          diag += w;
          if (nx >= 0 && nx < N && ny >= 0 && ny < N && nz >= 0 && nz < N)
            row.push_back({idx(nx, ny, nz), -w});
        };
        add(x - 1, y, z, 1.0); add(x + 1, y, z, 1.0);
        add(x, y - 1, z, 1.0); add(x, y + 1, z, 1.0);
        add(x, y, z - 1, 100.0); add(x, y, z + 1, 100.0);
        row.push_back({r, diag});
        std::sort(row.begin(), row.end());
        for (auto &kv : row) { ci.push_back(kv.first); val.push_back(kv.second); valf.push_back((float)kv.second); }
        rp[r + 1] = (int)ci.size();
      }
  int nnz = (int)ci.size();
  std::mt19937 rng(12345);
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<double> b(n);
  for (auto &v : b) v = nd(rng);

  int *d_rp, *d_ci; double *d_val, *d_b, *d_x;
  CHECK_CUDA(cudaMalloc(&d_rp, (n + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_ci, nnz * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_val, nnz * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(double)));
  CHECK_CUDA(cudaMemcpy(d_rp, rp.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_ci, ci.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_val, val.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));

  DevicePCGBuffers buf;
  buf.alloc(n);

  for (const std::string mode : {"int4_split", "nvfp4", "int4", "fp16", "fp64", "auto"}) {
    std::string cfg_str = make_vcycle_json("BITFSAI", mode, 0.72, 0);
    AMGX_config_handle cfg_f, cfg_d;
    AMGX_resources_handle rsrc_f, rsrc_d;
    AMGX_matrix_handle Af, Ad;
    AMGX_vector_handle vf_b, vf_x, vd_b, vd_x;
    AMGX_solver_handle sf, sd;

    CHECK_AMGX(AMGX_config_create(&cfg_f, cfg_str.c_str()));
    CHECK_AMGX(AMGX_config_create(&cfg_d, cfg_str.c_str()));
    CHECK_AMGX(AMGX_resources_create_simple(&rsrc_f, cfg_f));
    CHECK_AMGX(AMGX_resources_create_simple(&rsrc_d, cfg_d));
    CHECK_AMGX(AMGX_matrix_create(&Af, rsrc_f, AMGX_mode_dFFI));
    CHECK_AMGX(AMGX_matrix_create(&Ad, rsrc_d, AMGX_mode_dDDI));
    CHECK_AMGX(AMGX_vector_create(&vf_b, rsrc_f, AMGX_mode_dFFI));
    CHECK_AMGX(AMGX_vector_create(&vf_x, rsrc_f, AMGX_mode_dFFI));
    CHECK_AMGX(AMGX_vector_create(&vd_b, rsrc_d, AMGX_mode_dDDI));
    CHECK_AMGX(AMGX_vector_create(&vd_x, rsrc_d, AMGX_mode_dDDI));
    CHECK_AMGX(AMGX_solver_create(&sf, rsrc_f, AMGX_mode_dFFI, cfg_f));
    CHECK_AMGX(AMGX_solver_create(&sd, rsrc_d, AMGX_mode_dDDI, cfg_d));

    CHECK_AMGX(AMGX_matrix_upload_all(Af, n, nnz, 1, 1, rp.data(), ci.data(), valf.data(), nullptr));
    CHECK_AMGX(AMGX_matrix_upload_all(Ad, n, nnz, 1, 1, rp.data(), ci.data(), val.data(), nullptr));
    CHECK_AMGX(AMGX_solver_setup(sf, Af));
    CHECK_AMGX(AMGX_solver_setup(sd, Ad));

    auto res = solve_mixed_pcg_with_fallback(
        n, d_rp, d_ci, d_val, d_b, d_x, sf, vf_b, vf_x, sd, vd_b, vd_x, buf, 1e-8, 200, false);
    std::printf("[SELF-TEST] mode=%-10s iters=%3d recovered=%d rel=%.3e ms=%.2f\n",
                mode.c_str(), res.iters, (int)res.recovered, res.true_rel, res.solve_ms);
    if (!(res.true_rel <= 1e-8)) throw std::runtime_error("Self-test failed for mode " + mode);

    if (mode == "int4_split") {
      auto rec = solve_mixed_pcg_with_fallback(
          n, d_rp, d_ci, d_val, d_b, d_x, sf, vf_b, vf_x, sd, vd_b, vd_x, buf, 1e-8, 200, true);
      std::printf("[RECOVERY ] injected neg rTz -> trigger=%s recovered=%d iters=%3d rel=%.3e ms=%.2f\n",
                  rec.trigger.c_str(), (int)rec.recovered, rec.iters, rec.true_rel, rec.solve_ms);
      if (!rec.recovered || rec.trigger != "rz" || !(rec.true_rel <= 1e-8))
        throw std::runtime_error("Recovery self-test failed");
    }

    AMGX_solver_destroy(sf); AMGX_solver_destroy(sd);
    AMGX_vector_destroy(vf_b); AMGX_vector_destroy(vf_x);
    AMGX_vector_destroy(vd_b); AMGX_vector_destroy(vd_x);
    AMGX_matrix_destroy(Af); AMGX_matrix_destroy(Ad);
    AMGX_resources_destroy(rsrc_f); AMGX_resources_destroy(rsrc_d);
    AMGX_config_destroy(cfg_f); AMGX_config_destroy(cfg_d);
  }
  cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_val); cudaFree(d_b); cudaFree(d_x);
}

static void run_numerical_edge_case_tests() {
  auto run_bitfsai_2x2 = [](const std::string &name, const std::string &mode,
                            double a00, double a01, double a11,
                            double b0, double b1, double expected_z0, double expected_z1,
                            double rel_tol) {
    int rp[3] = {0, 2, 4};
    int ci[4] = {0, 1, 0, 1};
    double val[4] = {a00, a01, a01, a11};
    double bv[2] = {b0, b1};
    double xv[2] = {0.0, 0.0};

    std::string cfg_str =
        "{\"config_version\":2,\"determinism_flag\":0,\"solver\":{"
        "\"solver\":\"BITFSAI\",\"bitfsai_mode\":\"" + mode +
        "\",\"bitfsai_omega\":1.0,\"bitfsai_safe_omega\":0,\"max_iters\":1}}";
    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle vb, vx;
    AMGX_solver_handle solver;
    CHECK_AMGX(AMGX_config_create(&cfg, cfg_str.c_str()));
    CHECK_AMGX(AMGX_resources_create_simple(&rsrc, cfg));
    CHECK_AMGX(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI));
    CHECK_AMGX(AMGX_vector_create(&vb, rsrc, AMGX_mode_dDDI));
    CHECK_AMGX(AMGX_vector_create(&vx, rsrc, AMGX_mode_dDDI));
    CHECK_AMGX(AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg));

    CHECK_AMGX(AMGX_matrix_upload_all(A, 2, 4, 1, 1, rp, ci, val, nullptr));
    CHECK_AMGX(AMGX_solver_setup(solver, A));
    CHECK_AMGX(AMGX_vector_upload(vb, 2, 1, bv));
    CHECK_AMGX(AMGX_vector_set_zero(vx, 2, 1));
    CHECK_AMGX(AMGX_solver_solve(solver, vb, vx));
    CHECK_AMGX(AMGX_vector_download(vx, xv));

    double rz = b0 * xv[0] + b1 * xv[1];
    double err0 = std::fabs(xv[0] - expected_z0) / std::max(1e-300, std::fabs(expected_z0));
    double err1 = std::fabs(xv[1] - expected_z1) / std::max(1e-300, std::fabs(expected_z1));
    std::printf("[EDGE-CASE] %-24s z=(%.3e, %.3e) rTz=%.3e err=(%.2e, %.2e)\n",
                name.c_str(), xv[0], xv[1], rz, err0, err1);
    bool ok = std::isfinite(xv[0]) && std::isfinite(xv[1]) && (rz > 0.0) &&
              (err0 <= rel_tol) && (err1 <= rel_tol);
    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(vb);
    AMGX_vector_destroy(vx);
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(rsrc);
    AMGX_config_destroy(cfg);
    if (!ok) throw std::runtime_error("Edge-case test failed: " + name);
  };

  // 1. Tiny pivot 1e-28 (scale-free positivity, no 1e-24 cutoff)
  // Exact inverse of [[1e-28, 0.5e-28], [0.5e-28, 1e-28]] applied to [1, 2]:
  // det = 0.75e-56, inv*b = (1/0.75e-28)*[1 - 1, -0.5 + 2] = [0, 2e28];
  // use b = [1, 0] -> exact = [4/3 * 1e28, -2/3 * 1e28]
  run_bitfsai_2x2("tiny_pivot_1e-28", "int4_split",
                  1e-28, 0.5e-28, 1e-28,
                  1.0, 0.0, (4.0 / 3.0) * 1e28, -(2.0 / 3.0) * 1e28, 0.15);

  // 2. Small off-diagonal coupling (2^42, 2^21) must not drop coupling
  {
    double a00 = std::ldexp(1.0, 42);
    double a01 = std::ldexp(1.0, 21);
    double a11 = 2.0;
    // Exact inverse of [[a00, a01], [a01, 2]] has det = a00*2 - a01^2 = a00
    // inv = [[2/a00, -a01/a00], [-a01/a00, 1.0]]
    // Applied to [0, 1] -> [-2^-21, 1.0]
    run_bitfsai_2x2("small_coupling_2p42", "nvfp4",
                    a00, a01, a11,
                    0.0, 1.0, -a01 / a00, 1.0, 1e-3);
  }

  // 3. Sub-FP32 FP64 RHS (1e-48) via wide FP64 intermediate apply
  // inv([[4,-1],[-1,4]]) * [1e-48, -2e-48] = (1/15)*[2e-48, -7e-48]
  run_bitfsai_2x2("fp64_tiny_rhs_1e-48", "int4_split",
                  4.0, -1.0, 4.0,
                  1e-48, -2e-48, (2.0 / 15.0) * 1e-48, -(7.0 / 15.0) * 1e-48, 0.15);

  // 4. Ill-conditioned 2x2 patch (k=7000) triggers stored-energy Jacobi fallback
  // Row 0: g00 = 1/sqrt(a00); Row 1: falls back to diagonal Jacobi g11 = 1/sqrt(a11)
  {
    double k = 7000.0;
    double a00 = k * k + 1.0;
    double a01 = k * (k + 1.0) + 1.0;
    double a11 = (k + 1.0) * (k + 1.0) + 1.0;
    run_bitfsai_2x2("ill_cond_k7000_fallback", "nvfp4",
                    a00, a01, a11,
                    1.0, 1.0, 1.0 / a00, 1.0 / a11, 1e-4);
  }
}

int main(int argc, char **argv) {
  AMGX_initialize();
  try {
    run_self_test();
    run_numerical_edge_case_tests();
    std::printf("ALL BITFSAI AMGX PR SELF-TESTS PASSED.\n");
  } catch (const std::exception &e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    AMGX_finalize();
    return 1;
  }
  AMGX_finalize();
  return 0;
}
