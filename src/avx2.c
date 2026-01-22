
#include "cachepix.h"


#if defined(__AVX2__)
#include <immintrin.h>

int ppm_scale_avx2(PPM_ptr img_ptr, float scale, float bias)
{
    if (ppm_validate(img_ptr) < 0)
        return -1;

    const size_t row_bytes = img_ptr->width * 3;
    const __m256 vscale = _mm256_set1_ps(scale);
    const __m256 vbias  = _mm256_set1_ps(bias);
    const __m256 vzero  = _mm256_set1_ps(0.0f);
    const __m256 vmax   = _mm256_set1_ps(255.0f);

    for (size_t y = 0; y < img_ptr->height; ++y) {
        data_t row = img_ptr->data + y * img_ptr->stride;

        size_t i = 0;
        for (; i + 32 <= row_bytes; i += 32) {
            __m256i v = _mm256_loadu_si256((__m256i*)(row + i));

            // widen u8 → u16
            __m256i lo = _mm256_unpacklo_epi8(v, _mm256_setzero_si256());
            __m256i hi = _mm256_unpackhi_epi8(v, _mm256_setzero_si256());

            // u16 → u32
            __m256i lo32a = _mm256_unpacklo_epi16(lo, _mm256_setzero_si256());
            __m256i lo32b = _mm256_unpackhi_epi16(lo, _mm256_setzero_si256());
            __m256i hi32a = _mm256_unpacklo_epi16(hi, _mm256_setzero_si256());
            __m256i hi32b = _mm256_unpackhi_epi16(hi, _mm256_setzero_si256());

            // int → float
            __m256 f0 = _mm256_cvtepi32_ps(lo32a);
            __m256 f1 = _mm256_cvtepi32_ps(lo32b);
            __m256 f2 = _mm256_cvtepi32_ps(hi32a);
            __m256 f3 = _mm256_cvtepi32_ps(hi32b);

            // math
            f0 = _mm256_add_ps(_mm256_mul_ps(f0, vscale), vbias);
            f1 = _mm256_add_ps(_mm256_mul_ps(f1, vscale), vbias);
            f2 = _mm256_add_ps(_mm256_mul_ps(f2, vscale), vbias);
            f3 = _mm256_add_ps(_mm256_mul_ps(f3, vscale), vbias);

            // clamp
            f0 = _mm256_min_ps(_mm256_max_ps(f0, vzero), vmax);
            f1 = _mm256_min_ps(_mm256_max_ps(f1, vzero), vmax);
            f2 = _mm256_min_ps(_mm256_max_ps(f2, vzero), vmax);
            f3 = _mm256_min_ps(_mm256_max_ps(f3, vzero), vmax);

            // float → int
            lo32a = _mm256_cvtps_epi32(f0);
            lo32b = _mm256_cvtps_epi32(f1);
            hi32a = _mm256_cvtps_epi32(f2);
            hi32b = _mm256_cvtps_epi32(f3);

            // pack back
            lo = _mm256_packus_epi32(lo32a, lo32b);
            hi = _mm256_packus_epi32(hi32a, hi32b);
            v  = _mm256_packus_epi16(lo, hi);

            _mm256_storeu_si256((__m256i*)(row + i), v);
        }

        // scalar tail
        for (; i < row_bytes; ++i) {
            float v = row[i] * scale + bias;
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            row[i] = (uint8_t)v;
        }
    }

    return 0;
}

int ppm_rgb_to_grayscale_avx2(PPM_ptr dst_ptr, const PPM_ptr src_ptr) {
    if (ppm_validate(src_ptr) < 0 || ppm_validate(dst_ptr) < 0)
        return -1;

    const __m256 wR = _mm256_set1_ps(0.299f);
    const __m256 wG = _mm256_set1_ps(0.587f);
    const __m256 wB = _mm256_set1_ps(0.114f);

    for (size_t y = 0; y < src_ptr->height; ++y) {
        data_t s = src_ptr->data + y * src_ptr->stride;
        data_t d = dst_ptr->data + y * dst_ptr->stride;

        size_t x = 0;
        for (; x + 8 <= src_ptr->width; x += 8) {
            uint8_t r[8], g[8], b[8];

            for (int i = 0; i < 8; ++i) {
                r[i] = s[(x+i)*3 + 0];
                g[i] = s[(x+i)*3 + 1];
                b[i] = s[(x+i)*3 + 2];
            }

            __m256 fr = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)r)));
            __m256 fg = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)g)));
            __m256 fb = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)b)));

            __m256 yv = _mm256_add_ps(
                            _mm256_add_ps(_mm256_mul_ps(fr, wR),
                                          _mm256_mul_ps(fg, wG)),
                            _mm256_mul_ps(fb, wB));

            __m256i yi = _mm256_cvtps_epi32(yv);
            __m256i y8 = _mm256_packus_epi32(yi, yi);
            y8 = _mm256_packus_epi16(y8, y8);

            *(uint64_t*)(d + x) = _mm256_extract_epi64(y8, 0);
        }

        for (; x < src_ptr->width; ++x) {
            uint8_t R = s[x*3];
            uint8_t G = s[x*3+1];
            uint8_t B = s[x*3+2];
            d[x] = (uint8_t)(0.299f*R + 0.587f*G + 0.114f*B);
        }
    }

    return 0;
}

int ppm_convert_maxval_avx2(PPM_ptr img_ptr, uint16_t new_maxval)
{
    if (ppm_validate(img_ptr) < 0)
        return -1;

    float scale = (float)new_maxval / (float)img_ptr->maxval;
    __m256 vscale = _mm256_set1_ps(scale);
    __m256 vmax = _mm256_set1_ps((float)new_maxval);

    for (size_t y = 0; y < img_ptr->height; ++y) {
        data_t row = img_ptr->data + y * img_ptr->stride;

        size_t i = 0;
        for (; i + 32 <= img_ptr->width * 3; i += 32) {
            __m256i v = _mm256_loadu_si256((__m256i*)(row + i));

            __m256i lo = _mm256_unpacklo_epi8(v, _mm256_setzero_si256());
            __m256i hi = _mm256_unpackhi_epi8(v, _mm256_setzero_si256());

            __m256 f0 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(lo, _mm256_setzero_si256()));
            __m256 f1 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(lo, _mm256_setzero_si256()));
            __m256 f2 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(hi, _mm256_setzero_si256()));
            __m256 f3 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(hi, _mm256_setzero_si256()));

            f0 = _mm256_min_ps(_mm256_mul_ps(f0, vscale), vmax);
            f1 = _mm256_min_ps(_mm256_mul_ps(f1, vscale), vmax);
            f2 = _mm256_min_ps(_mm256_mul_ps(f2, vscale), vmax);
            f3 = _mm256_min_ps(_mm256_mul_ps(f3, vscale), vmax);

            __m256i lo32a = _mm256_cvtps_epi32(f0);
            __m256i lo32b = _mm256_cvtps_epi32(f1);
            __m256i hi32a = _mm256_cvtps_epi32(f2);
            __m256i hi32b = _mm256_cvtps_epi32(f3);

            lo = _mm256_packus_epi32(lo32a, lo32b);
            hi = _mm256_packus_epi32(hi32a, hi32b);
            v  = _mm256_packus_epi16(lo, hi);

            _mm256_storeu_si256((__m256i*)(row + i), v);
        }

        for (; i < img_ptr->width * 3; ++i)
            row[i] = (uint8_t)(row[i] * scale);
    }

    img_ptr->maxval = new_maxval;
    return 0;
}
#endif
