# RV-Sparse Coding Challenge

Implements `sparse_multiply`: converts a dense row-major matrix to CSR format and computes the sparse matrix-vector product `y = A * x` with zero dynamic allocation.

Non-zero elements are identified using `fabs(val) > DBL_EPSILON` rather than `== 0`, avoiding false positives from floating-point rounding where a value may be non-zero but indistinguishable from zero at double precision.

`challenge_rvv.c` extends this with a RISC-V Vector (RVV) optimised SpMV kernel using gather-multiply-reduce intrinsics, falling back to scalar on non-RVV platforms via `#ifdef __riscv_vector`.

---

## challenge.c — Scalar Solution

```
gcc challenge.c -lm -o run
./run
```

> **Note:** The challenge documentation shows `gcc -lm -o run challenge.c`. This can fail on some linkers because the linker processes arguments left to right — it searches `libm` before seeing the references in `challenge.o`. Placing `-lm` after the source file is more portable.

---

## challenge_rvv.c — RVV Optimised Solution

### Dependencies

```
sudo apt install qemu-user gcc-riscv64-linux-gnu
```

### Build

Scalar RISC-V binary (no RVV):
```
riscv64-linux-gnu-gcc -march=rv64gc -mabi=lp64d -O2 -static challenge_rvv.c -lm -o run_scalar_riscv
```

RVV binary:
```
riscv64-linux-gnu-gcc -march=rv64gcv -mabi=lp64d -O2 -static challenge_rvv.c -lm -o run_rvv_riscv
```

### Run on QEMU

```
qemu-riscv64 -cpu max ./run_rvv_riscv
```

> **Important:** QEMU emulates instructions sequentially and does not model pipeline parallelism. Running on QEMU confirms correctness only — actual speedups from RVV vectorisation require real RISC-V hardware (e.g. SiFive, StarFive VisionFive 2) or a cycle-accurate simulator such as Spike.

---

## Tested On

| Component | Version |
|---|---|
| OS | Ubuntu 6.17.0-23-generic x86_64 |
| CPU | AMD Ryzen 7 7735U |
| GCC (host) | Ubuntu 15.2.0 |
| GCC (RISC-V cross) | riscv64-linux-gnu-gcc 15.2.0 |
| QEMU | 10.1.0 |
