#include <emmintrin.h> // SSE2
#include <smmintrin.h>

#include "cachepix.h"
#include "internal.h"

int ppm_scale_sse2(PPM_ptr img_ptr, float scale, float bias) {
    if (ppm_validate(img_ptr) < 0)
        return -1;

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

int ppm_rgb_to_grayscale_sse2(PPM_ptr img)
{
    if (ppm_validate(img_ptr) < 0)
        return -1;

    const size_t row_bytes = img->width * 3;

    const __m128 wr = _mm_set1_ps(0.299f);
    const __m128 wg = _mm_set1_ps(0.587f);
    const __m128 wb = _mm_set1_ps(0.114f);

    for (size_t y = 0; y < img->height; ++y) {
        uint8_t *row = img->data + y * img->stride;

        for (size_t x = 0; x + 12 <= row_bytes; x += 12) {
            // load 4 pixels (12 bytes)
            uint8_t *p = row + x;

            float r[4] = { p[0], p[3], p[6], p[9] };
            float g[4] = { p[1], p[4], p[7], p[10] };
            float b[4] = { p[2], p[5], p[8], p[11] };

            __m128 vr = _mm_loadu_ps(r);
            __m128 vg = _mm_loadu_ps(g);
            __m128 vb = _mm_loadu_ps(b);

            __m128 yv = _mm_add_ps(
                            _mm_add_ps(_mm_mul_ps(vr, wr),
                                       _mm_mul_ps(vg, wg)),
                            _mm_mul_ps(vb, wb));

            __m128i yi = _mm_cvtps_epi32(yv);

            uint32_t yout[4];
            _mm_storeu_si128((__m128i *)yout, yi);

            for (int i = 0; i < 4; ++i) {
                uint8_t g = (uint8_t)yout[i];
                p[i*3+0] = g;
                p[i*3+1] = g;
                p[i*3+2] = g;
            }
        }
    }

    return 0;
}

