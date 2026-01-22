# cachepix

`cachepix` is a high-performance C library for loading, storing, and processing **PPM (Portable Pixmap)** images, with a strong focus on **data layout, cache efficiency, and SIMD optimization**.

The project is designed as a systems-level exercise in performance engineering and serves as a demonstration of low-level optimization techniques relevant to kernel and systems development.

---

## Features

- Full support for **PPM (P6)** image formats
- Robust parsing and validation of PPM headers
- Explicit handling of `maxval` (8-bit and 16-bit samples)
- Row-stride–aware image layout for cache friendliness
- Scalar reference implementations
- SIMD-accelerated implementations:
  - **SSE2** (x86)
  - **AVX2** (x86)
  - **NEON** (ARM)
- Compile-time SIMD selection
- No external dependencies other than libc

---

## Design Goals

- **Performance first**: predictable memory access patterns, cache-aligned rows, SIMD-friendly data layout
- **Explicitness**: no hidden allocations or implicit conversions
- **Portability**: scalar fallback always available
- **Clarity**: SIMD implementations are separated by architecture
- **Correctness**: careful handling of endianness, overflow, and stride

---

## SIMD Architecture

Each optimized backend implements the same internal operations with architecture-specific intrinsics.  
Only one backend is selected at **compile time**, avoiding runtime CPUID overhead and illegal instruction risks.

Priority order:
1. AVX2 (if enabled)
2. SSE2 (if enabled)
3. NEON (ARM)
4. Scalar fallback

The public API remains identical regardless of backend.

---

## Public API Overview

```c
PPM_ptr load_ppm_image(const char *filename);
int save_ppm_image(PPM_ptr img, const char *filename, int force);
void free_ppm_image(PPM_ptr img);

int ppm_apply_scalar(PPM_ptr img, float scale, float bias);
int ppm_rgb_to_grayscale(const PPM_ptr src, PPM_ptr dst);
int ppm_convert_maxval(PPM_ptr img, uint16_t new_maxval);

int ppm_validate(const PPM_ptr img);
int ppm_is_contiguous(const PPM_ptr img);
```

All functions return 0 on success and a negative number on error.

---

## Memory layout

* Pixel data is stored in row-major order

* Each row begins at a stride aligned to a cache-friendly boundary

* Padding bytes may exist between rows

* SIMD code operates only on valid pixel data, never padding

This layout improves cache utilization and enables aligned SIMD loads.

---

## PPM Details

* Supports both 8-bit and 16-bit channel depths

* 16-bit samples are handled explicitly as big-endian, per PPM specification

* maxval scaling preserves relative intensity

* Comments and whitespace are handled according to the PPM spec

---

## Building

```sh
make 
# or
make ARCH=[x86, x86_avx2, arm]
```

---

## Usage Example

```c
PPM_ptr img = load_ppm_image("input.ppm");
ppm_apply_scalar(img, 1.2f, 0.0f);
save_ppm_image(img, "output.ppm", 1);
free_ppm_image(img);
```

---

## Error Handling

* All public functions perform internal validation

* Invalid images or unsupported operations return errors

* No undefined behavior is relied upon
