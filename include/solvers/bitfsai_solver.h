// SPDX-FileCopyrightText: 2026 Brett Bonner
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <solvers/solver.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace amgx
{

template <class T_Config> class BitFSAISolver;

// Zero-Duplicate-Index Sub-Byte Factorized Sparse Approximate Inverse (BitFSAI) Smoother.
//
// Factor encodings (configured via `bitfsai_mode`):
//   int4_split (default): Split-scale symmetric signed INT4 off-diagonals + FP32 diagonal (0.5 B/nnz)
//   fp4_e2m1            : Split-scale signed FP4 (E2M1) off-diagonals + FP32 diagonal (0.5 B/nnz)
//   fp16                : Split-scale FP16 off-diagonals + FP32 diagonal (2.0 B/nnz)
//   fp64                : Uncompressed FP64 reference (8.0 B/nnz)
//
// Supported AMGX config parameters (registered in src/core.cu):
//   bitfsai_mode       = "int4_split" | "fp4_e2m1" | "fp16" | "fp64"
//   bitfsai_omega      = damping factor (> 0, default 0.72)
//   bitfsai_max_patch  = max local Cholesky patch size 1..16 (0 = auto-adaptive 8/12)
//   bitfsai_safe_omega = 1 to cap damping via conservative upward-rounded Gershgorin bound 1 / U_upper
//   bitfsai_reproject  = 1 to apply post-quantization 2x2 A-energy optimal re-projection (default 1)
template <class T_Config>
class BitFSAISolver_Base : public Solver<T_Config>
{
    public:
        typedef Solver<T_Config> Base;
        typedef T_Config TConfig;
        static const AMGX_VecPrecision vecPrec = TConfig::vecPrec;
        static const AMGX_MatPrecision matPrec = TConfig::matPrec;
        static const AMGX_IndPrecision indPrec = TConfig::indPrec;
        typedef typename T_Config::MatPrec ValueTypeA;
        typedef typename T_Config::VecPrec ValueTypeB;
        typedef Vector<T_Config> VVector;

        enum class Mode : int
        {
            Int4Split = 0,
            Fp4E2m1   = 1,
            Fp16      = 2,
            Fp64      = 3
        };

    protected:
        Matrix<T_Config> *m_explicit_A = nullptr;
        Mode   m_mode = Mode::Int4Split;
        int    m_max_patch_cfg = 0;      // 0 = auto-adaptive (8 for avg_deg > 22, else 12)
        double m_omega_cfg = 0.72;       // Configured base damping
        double m_omega = 0.72;           // Effective damping (may be capped by safe_omega)
        bool   m_use_safe_omega = false;
        bool   m_use_reproject = true;
        bool   m_wide_apply = false;

        int  m_n = 0;
        int  m_nnz = 0;
        int  m_alloc_mode = -1;
        bool m_alloc_wide = false;

        void free_buffers();

        // Zero-duplicate-index persistent GPU factor storage (aligned 1-to-1 with A's CSR):
        float*    d_g_diag = nullptr;      // FP32 diagonal g_ii (Int4Split, Fp4E2m1, Fp16)
        float*    d_g_off_scale = nullptr; // FP32 per-row scale s_i (Int4Split, Fp4E2m1, Fp16)
        uint8_t*  d_q_packed = nullptr;    // 4-bit packed nibbles (ceil(nnz/2) B, Int4Split, Fp4E2m1)
        __half*   d_g_f16 = nullptr;       // FP16 weights (2*nnz B, Fp16)

        // Uncompressed FP64 reference buffers (Fp64 only):
        double*   d_g_diag_f64 = nullptr;  // FP64 diagonal g_ii
        double*   d_g_f64 = nullptr;       // FP64 off-diagonal weights (8*nnz B)

        // Apply workspace buffers:
        //   !m_wide_apply: (n + 1) * sizeof(double) bytes (4 B/row float y[n] overlaid with
        //                  fallback double dtmp[n], plus trailing int oor_flag at dtmp + n).
        //    m_wide_apply: n * sizeof(double) bytes (double t_f64[n]).
        void*       d_work = nullptr;
        ValueTypeB* d_res = nullptr;       // Fused residual buffer r = b - A*x

        virtual void setup_device(const Matrix<T_Config> &A, bool reuse) = 0;
        virtual void smooth_device(const Matrix<T_Config> &A, const VVector &b, VVector &x, bool xIsZero) = 0;

    public:
        BitFSAISolver_Base(AMG_Config &cfg, const std::string &cfg_scope);
        ~BitFSAISolver_Base();

        bool is_residual_needed() const { return false; }
        void solver_setup(bool reuse_matrix_structure);
        bool isColoringNeeded() const { return false; }
        bool getReorderColsByColorDesired() const { return false; }
        bool getInsertDiagonalDesired() const { return false; }
        void solve_init(VVector &b, VVector &x, bool xIsZero) {}
        AMGX_STATUS solve_iteration(VVector &b, VVector &x, bool xIsZero);
        void solve_finalize(VVector &b, VVector &x) {}
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class BitFSAISolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
    : public BitFSAISolver_Base<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef Vector<TConfig_h> VVector;
        BitFSAISolver(AMG_Config &cfg, const std::string &cfg_scope)
            : BitFSAISolver_Base<TConfig_h>(cfg, cfg_scope) {}
    protected:
        void setup_device(const Matrix_h &A, bool reuse) override
        {
            FatalError("BitFSAISolver does not support host execution mode", AMGX_ERR_NOT_IMPLEMENTED);
        }
        void smooth_device(const Matrix_h &A, const VVector &b, VVector &x, bool xIsZero) override
        {
            FatalError("BitFSAISolver does not support host execution mode", AMGX_ERR_NOT_IMPLEMENTED);
        }
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class BitFSAISolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
    : public BitFSAISolver_Base<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef Vector<TConfig_d> VVector;
        BitFSAISolver(AMG_Config &cfg, const std::string &cfg_scope)
            : BitFSAISolver_Base<TConfig_d>(cfg, cfg_scope) {}
    protected:
        void setup_device(const Matrix_d &A, bool reuse) override;
        void smooth_device(const Matrix_d &A, const VVector &b, VVector &x, bool xIsZero) override;
};

template <class T_Config>
class BitFSAISolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng)
        {
            return new BitFSAISolver<T_Config>(cfg, cfg_scope);
        }
};

} // namespace amgx
