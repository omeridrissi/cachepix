
#include "cachepix.h"

#if defined(__SSE2__)
#include <emmintrin.h> // SSE2
#include <smmintrin.h>


int ppm_scale_sse2(PPM_ptr img_ptr, float scale, float bias) {
    if (ppm_validate(img_ptr) < 0)
        return -1;

    const size_t row_bytes = img_ptr->width*3;
    const __m128 vscale = _mm_set1_ps(scale);
    const __m128 vbias = _mm_set1_ps(bias);

    for (size_t y = 0; y < img_ptr->height; ++y) {
        data_t row = img_ptr->data + y * img_ptr->stride;
        size_t x = 0;

        for (; x + 16 <= row_bytes; x+=16) {
            __m128i bytes = _mm_load_si128((__m128i *)(row+x));

            // unpack u8 -> u16
            __m128i lo = _mm_unpacklo_epi8(bytes, _mm_setzero_si128());
            __m128i hi = _mm_unpackhi_epi8(bytes, _mm_setzero_si128());

            // convert to float
            __m128 flo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(lo, _mm_setzero_si128()));
            __m128 fhi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(hi, _mm_setzero_si128()));

            flo = _mm_add_ps(_mm_mul_ps(flo, vscale), vbias);
            fhi = _mm_add_ps(_mm_mul_ps(fhi, vscale), vbias);

            __m128i ilo = _mm_cvtps_epi32(flo);
            __m128i ihi = _mm_cvtps_epi32(fhi);

            __m128i packed16 = _mm_packus_epi32(ilo, ihi);
            __m128i packed8 = _mm_packus_epi16(packed16, packed16);

            _mm_store_si128((__m128i *)(row + x), packed8);
        }

        for (; x < row_bytes; ++x) {
            row[x] = (uint8_t)(row[x] * scale + bias);
        }

    }

    return 0;
}

int ppm_convert_maxval_sse2(PPM_ptr img_ptr, uint16_t new_maxval) {
    if (ppm_validate(img_ptr) < 0)
        return -1;
    double scale = (double)new_maxval / (double)img_ptr->maxval;
    ppm_scale_sse2(img_ptr, scale, 0.0);
    img_ptr->maxval = new_maxval;
    return 0;
}

int ppm_rgb_to_grayscale_sse2(PPM_ptr dst_ptr, const PPM_ptr src_ptr)
{
    if (ppm_validate(dst_ptr) < 0 || ppm_validate(src_ptr) < 0)
        return -1;

    //const size_t row_bytes = src_ptr->width * 3;

    // fixed-point weights (Q8) to avoid float math
    const __m128i wR = _mm_set1_epi16(77);   // 0.299 * 256 ≈ 77
    const __m128i wG = _mm_set1_epi16(150);  // 0.587 * 256 ≈ 150
    const __m128i wB = _mm_set1_epi16(29);   // 0.114 * 256 ≈ 29

    for (size_t y = 0; y < src_ptr->height; ++y) {
        data_t srow = src_ptr->data + y * src_ptr->stride;
        data_t drow = dst_ptr->data + y * dst_ptr->stride;

        size_t x = 0;

        // Process 16 pixels at a time (16 * 3 = 48 bytes)
        for (; x + 16 <= src_ptr->width; x += 16) {
            // Load 48 bytes (16 RGB pixels) in three 16-byte chunks
            __m128i r_chunk, g_chunk, b_chunk;
            uint8_t rvals[16], gvals[16], bvals[16];

            for (int i = 0; i < 16; ++i) {
                rvals[i] = srow[(x + i) * 3 + 0];
                gvals[i] = srow[(x + i) * 3 + 1];
                bvals[i] = srow[(x + i) * 3 + 2];
            }

            r_chunk = _mm_loadu_si128((__m128i*)rvals);
            g_chunk = _mm_loadu_si128((__m128i*)gvals);
            b_chunk = _mm_loadu_si128((__m128i*)bvals);

            // unpack to 16-bit integers
            __m128i r_lo = _mm_unpacklo_epi8(r_chunk, _mm_setzero_si128());
            __m128i r_hi = _mm_unpackhi_epi8(r_chunk, _mm_setzero_si128());
            __m128i g_lo = _mm_unpacklo_epi8(g_chunk, _mm_setzero_si128());
            __m128i g_hi = _mm_unpackhi_epi8(g_chunk, _mm_setzero_si128());
            __m128i b_lo = _mm_unpacklo_epi8(b_chunk, _mm_setzero_si128());
            __m128i b_hi = _mm_unpackhi_epi8(b_chunk, _mm_setzero_si128());

            // multiply by fixed-point weights
            __m128i gray_lo = _mm_add_epi16(
                                  _mm_add_epi16(_mm_mullo_epi16(r_lo, wR),
                                                _mm_mullo_epi16(g_lo, wG)),
                                  _mm_mullo_epi16(b_lo, wB));
            __m128i gray_hi = _mm_add_epi16(
                                  _mm_add_epi16(_mm_mullo_epi16(r_hi, wR),
                                                _mm_mullo_epi16(g_hi, wG)),
                                  _mm_mullo_epi16(b_hi, wB));

            // shift back down (divide by 256)
            gray_lo = _mm_srli_epi16(gray_lo, 8);
            gray_hi = _mm_srli_epi16(gray_hi, 8);

            // pack back to 8-bit
            __m128i gray = _mm_packus_epi16(gray_lo, gray_hi);

            // store to dst_ptr row
            _mm_storeu_si128((__m128i*)(drow + x), gray);
        }

        // scalar tail
        for (; x < src_ptr->width; ++x) {
            uint8_t R = srow[x*3 + 0];
            uint8_t G = srow[x*3 + 1];
            uint8_t B = srow[x*3 + 2];
            drow[x] = (uint8_t)((77*R + 150*G + 29*B) >> 8);
        }
    }

    return 0;
}

#endif
