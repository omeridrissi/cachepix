#include "cachepix.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>

int ppm_scale_neon(PPM_ptr img_ptr, float scale, float bias)
{
    if (ppm_validate(img_ptr) < 0)
        return -1;

    const size_t row_bytes = img_ptr->width * 3;

    float32x4_t vscale = vdupq_n_f32(scale);
    float32x4_t vbias  = vdupq_n_f32(bias);
    float32x4_t vzero  = vdupq_n_f32(0.0f);
    float32x4_t vmax   = vdupq_n_f32(255.0f);

    for (size_t y = 0; y < img_ptr->height; ++y) {
        uint8_t *row = img_ptr->data + y * img_ptr->stride;

        size_t i = 0;
        for (; i + 16 <= row_bytes; i += 16) {
            uint8x16_t v = vld1q_u8(row + i);

            uint16x8_t lo16 = vmovl_u8(vget_low_u8(v));
            uint16x8_t hi16 = vmovl_u8(vget_high_u8(v));

            uint32x4_t lo32a = vmovl_u16(vget_low_u16(lo16));
            uint32x4_t lo32b = vmovl_u16(vget_high_u16(lo16));
            uint32x4_t hi32a = vmovl_u16(vget_low_u16(hi16));
            uint32x4_t hi32b = vmovl_u16(vget_high_u16(hi16));

            float32x4_t f0 = vcvtq_f32_u32(lo32a);
            float32x4_t f1 = vcvtq_f32_u32(lo32b);
            float32x4_t f2 = vcvtq_f32_u32(hi32a);
            float32x4_t f3 = vcvtq_f32_u32(hi32b);

            f0 = vaddq_f32(vmulq_f32(f0, vscale), vbias);
            f1 = vaddq_f32(vmulq_f32(f1, vscale), vbias);
            f2 = vaddq_f32(vmulq_f32(f2, vscale), vbias);
            f3 = vaddq_f32(vmulq_f32(f3, vscale), vbias);

            f0 = vminq_f32(vmaxq_f32(f0, vzero), vmax);
            f1 = vminq_f32(vmaxq_f32(f1, vzero), vmax);
            f2 = vminq_f32(vmaxq_f32(f2, vzero), vmax);
            f3 = vminq_f32(vmaxq_f32(f3, vzero), vmax);

            lo32a = vcvtq_u32_f32(f0);
            lo32b = vcvtq_u32_f32(f1);
            hi32a = vcvtq_u32_f32(f2);
            hi32b = vcvtq_u32_f32(f3);

            lo16 = vcombine_u16(vmovn_u32(lo32a), vmovn_u32(lo32b));
            hi16 = vcombine_u16(vmovn_u32(hi32a), vmovn_u32(hi32b));

            v = vcombine_u8(vmovn_u16(lo16), vmovn_u16(hi16));

            vst1q_u8(row + i, v);
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

int ppm_rgb_to_grayscale_neon(PPM_ptr dst, const PPM_ptr src)
{
    if (ppm_validate(src) < 0 || ppm_validate(dst) < 0) 
        return -1;

    float32x4_t wR = vdupq_n_f32(0.299f);
    float32x4_t wG = vdupq_n_f32(0.587f);
    float32x4_t wB = vdupq_n_f32(0.114f);

    for (size_t y = 0; y < src->height; ++y) {
        uint8_t *s = src->data + y * src->stride;
        uint8_t *d = dst->data + y * dst->stride;

        size_t x = 0;
        for (; x + 8 <= src->width; x += 8) {
            uint8_t r[8], g[8], b[8];

            for (int i = 0; i < 8; ++i) {
                r[i] = s[(x+i)*3 + 0];
                g[i] = s[(x+i)*3 + 1];
                b[i] = s[(x+i)*3 + 2];
            }

            uint16x8_t r16 = vmovl_u8(vld1_u8(r));
            uint16x8_t g16 = vmovl_u8(vld1_u8(g));
            uint16x8_t b16 = vmovl_u8(vld1_u8(b));

            float32x4_t r0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(r16)));
            float32x4_t r1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(r16)));
            float32x4_t g0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(g16)));
            float32x4_t g1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(g16)));
            float32x4_t b0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b16)));
            float32x4_t b1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b16)));

            float32x4_t y0 = vaddq_f32(
                                vaddq_f32(vmulq_f32(r0, wR),
                                          vmulq_f32(g0, wG)),
                                vmulq_f32(b0, wB));

            float32x4_t y1 = vaddq_f32(
                                vaddq_f32(vmulq_f32(r1, wR),
                                          vmulq_f32(g1, wG)),
                                vmulq_f32(b1, wB));

            uint16x8_t y16 = vcombine_u16(
                vmovn_u32(vcvtq_u32_f32(y0)),
                vmovn_u32(vcvtq_u32_f32(y1))
            );

            vst1_u8(d + x, vmovn_u16(y16));
        }

        for (; x < src->width; ++x) {
            uint8_t R = s[x*3];
            uint8_t G = s[x*3+1];
            uint8_t B = s[x*3+2];
            d[x] = (uint8_t)(0.299f*R + 0.587f*G + 0.114f*B);
        }
    }

    return 0;
}

int ppm_convert_maxval_neon(PPM_ptr img_ptr, uint16_t new_maxval)
{
    if (ppm_validate(img_ptr) < 0) 
        return -1;

    float scale = (float)new_maxval / (float)img_ptr->maxval;
    float32x4_t vscale = vdupq_n_f32(scale);
    float32x4_t vmax   = vdupq_n_f32((float)new_maxval);

    for (size_t y = 0; y < img_ptr->height; ++y) {
        uint8_t *row = img_ptr->data + y * img_ptr->stride;

        size_t i = 0;
        for (; i + 16 <= img_ptr->width * 3; i += 16) {
            uint8x16_t v = vld1q_u8(row + i);

            uint16x8_t lo16 = vmovl_u8(vget_low_u8(v));
            uint16x8_t hi16 = vmovl_u8(vget_high_u8(v));

            float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
            float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16)));
            float32x4_t f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
            float32x4_t f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16)));

            f0 = vminq_f32(vmulq_f32(f0, vscale), vmax);
            f1 = vminq_f32(vmulq_f32(f1, vscale), vmax);
            f2 = vminq_f32(vmulq_f32(f2, vscale), vmax);
            f3 = vminq_f32(vmulq_f32(f3, vscale), vmax);

            lo16 = vcombine_u16(
                vmovn_u32(vcvtq_u32_f32(f0)),
                vmovn_u32(vcvtq_u32_f32(f1))
            );
            hi16 = vcombine_u16(
                vmovn_u32(vcvtq_u32_f32(f2)),
                vmovn_u32(vcvtq_u32_f32(f3))
            );

            v = vcombine_u8(vmovn_u16(lo16), vmovn_u16(hi16));
            vst1q_u8(row + i, v);
        }

        for (; i < img_ptr->width * 3; ++i)
            row[i] = (uint8_t)(row[i] * scale);
    }

    img_ptr->maxval = new_maxval;
    return 0;
}

#endif
