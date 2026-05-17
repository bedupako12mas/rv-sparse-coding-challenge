#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

// Guard prevents x86 compilers from failing on a missing header
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

// =========================================================
// FUNCTION PROTOTYPE
// =========================================================
void sparse__multiply(
    int rows,
    int cols,
    const double* A,
    const double* x,
    int* out_nnz,
    double* values,
    int* col_indices,
    int* row_ptrs,
    double* y
);

// =========================================================
// TODO: USER IMPLEMENTATION
// =========================================================
void sparse_multiply(
    int rows, int cols, const double* A, const double* x,
    int* out_nnz, double* values, int* col_indices, int* row_ptrs,
    double* y
) {
    // Stage 1: CSR conversion (scalar)
    *row_ptrs = 0;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            if (fabs(A[cols * row + col]) > DBL_EPSILON) {
                col_indices[*out_nnz] = col;
                values[*out_nnz] = A[cols * row + col];
                ++(*out_nnz);
            }
        }
        row_ptrs[row + 1] = *out_nnz;
    }

    // Stage 2: SpMV — gather-multiply-reduce per row
#ifdef __riscv_vector
    for (int row = 0; row < rows; row++) {
        int ptr   = row_ptrs[row];
        int count = row_ptrs[row + 1] - ptr;

        // vsum[0] accumulates the dot product across while iterations
        vfloat64m1_t vsum = __riscv_vfmv_s_f_f64m1(0.0, 1);

        while (count > 0) {
            // Hardware grants vl <= count; tail handled automatically on last iteration
            size_t vl = __riscv_vsetvl_e64m1((size_t)count);

            vfloat64m1_t vval = __riscv_vle64_v_f64m1(values + ptr, vl);
            vuint32mf2_t vidx = __riscv_vle32_v_u32mf2((const uint32_t *)(col_indices + ptr), vl);

            // vluxei32 takes byte offsets, not element indices: idx << 3 == idx * sizeof(double)
            // Index LMUL is mf2 because EEW_index(32) is half of EEW_data(64)
            vuint32mf2_t vbyte_idx = __riscv_vsll_vx_u32mf2(vidx, 3, vl);
            vfloat64m1_t vx        = __riscv_vluxei32_v_f64m1(x, vbyte_idx, vl);

            vfloat64m1_t vprod = __riscv_vfmul_vv_f64m1(vval, vx, vl);
            vsum = __riscv_vfredusum_vs_f64m1_f64m1(vprod, vsum, vl);

            ptr   += (int)vl;
            count -= (int)vl;
        }

        y[row] = __riscv_vfmv_f_s_f64m1_f64(vsum);
    }
#else
    // Scalar fallback for non-RVV platforms
    for (int row = 0; row < rows; row++) {
        y[row] = 0.0;
        for (int iter = row_ptrs[row]; iter < row_ptrs[row + 1]; ++iter) {
            y[row] += values[iter] * x[col_indices[iter]];
        }
    }
#endif
}

// =========================================================
// TEST HARNESS
// =========================================================
int main(void) {
    srand(time(NULL));

    const int num_iterations = 100;
    int passed_count = 0;

    for (int iter = 0; iter < num_iterations; ++iter) {
        int rows = rand() % 41 + 5;
        int cols = rand() % 41 + 5;
        double density = 0.05 + (rand() / (double) RAND_MAX) * 0.35;

        size_t mat_sz = (size_t) rows * cols;

        double* A = calloc(mat_sz, sizeof(double));
        for (size_t i = 0; i < mat_sz; ++i) {
            if (((double) rand() / RAND_MAX) < density) {
                A[i] = ((double) rand() / RAND_MAX) * 20.0 - 10.0;
            }
        }

        double* values     = malloc(mat_sz * sizeof(double));
        int*    col_indices = malloc(mat_sz * sizeof(int));
        int*    row_ptrs   = malloc((rows + 1) * sizeof(int));
        double* x          = malloc(cols * sizeof(double));
        double* y_user     = malloc(rows * sizeof(double));
        double* y_ref      = calloc(rows, sizeof(double));
        int     out_nnz    = 0;

        for (int i = 0; i < cols; ++i)
            x[i] = ((double) rand() / RAND_MAX) * 20.0 - 10.0;

        for (int i = 0; i < rows; ++i) {
            double sum = 0.0;
            for (int j = 0; j < cols; ++j)
                sum += A[i * cols + j] * x[j];
            y_ref[i] = sum;
        }

        sparse_multiply(rows, cols, A, x, &out_nnz, values, col_indices, row_ptrs, y_user);

        double max_err = 0.0;
        int passed = 1;
        for (int i = 0; i < rows; ++i) {
            double diff = fabs(y_user[i] - y_ref[i]);
            double tol  = 1e-7 + 1e-7 * fabs(y_ref[i]);
            if (diff > tol) {
                max_err = fmax(max_err, diff);
                passed  = 0;
            }
        }

        if (passed) passed_count++;

        printf(
            "Iter %2d [%3dx%3d, density=%.2f, nnz=%4d]: %s (Max error: %.2e)\n",
            iter, rows, cols, density, out_nnz, passed ? "PASS" : "FAIL", max_err
        );

        free(A); free(values); free(col_indices);
        free(row_ptrs); free(x); free(y_user); free(y_ref);
    }

    printf(
        "\n%s (%d/%d iterations passed)\n",
        passed_count == num_iterations ? "All tests passed!" : "Some tests failed.",
        passed_count, num_iterations
    );

    return passed_count == num_iterations ? 0 : 1;
}
