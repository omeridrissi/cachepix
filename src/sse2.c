#include <emmintrin.h> // SSE2
#include <smmintrin.h>

#include "cachepix.h"
#include "internal.h"

int ppm_scale_sse2(PPM_ptr img_ptr, float scale, float bias) {
    const size_t row_bytes = img_ptr->width * 3;
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
            __m128 fhi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(lo, _mm_setzero_si128()));

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
}

int ppm_convert_maxval_sse2(PPM_ptr img_ptr, uint16_t new_maxval) {
    double scale = (double)new_maxval / (double)img_ptr->maxval;
    ppm_scale_sse2(img_ptr, scale, 0.0);
    img_ptr->maxval = new_maxval;
}
