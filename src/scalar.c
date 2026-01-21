#include <stdlib.h>

#include "cachepix.h"

#include "internal.h"

int ppm_convert_maxval_scalar(PPM_ptr img_ptr, uint16_t new_maxval) {

    if (ppm_validate(img_ptr) < 0)
        return -1;

    uint16_t old_maxval = img_ptr->maxval;

    if (new_maxval == old_maxval)
        return 0;

    size_t old_bpc = (old_maxval <= 255) ? 1 : 2;    
    size_t new_bpc = (new_maxval <= 255) ? 1 : 2;    

    size_t new_row_bytes = img_ptr->width * 3 * new_bpc;

    size_t new_stride = (new_row_bytes + 63) & ~((size_t)63);

    data_t new_data = (data_t)malloc(new_stride*img_ptr->width);
    if (!new_data)
        return -1;

    for (size_t y = 0; y < img_ptr->height; ++y) {
        data_t src_row = img_ptr->data + y*img_ptr->stride;
        data_t dst_row = new_data + y*new_stride;

        if (old_bpc == 1 && new_bpc == 1) {
            /* 8 -> 8 */
            for (size_t i = 0; i < img_ptr->width*3; ++i) {
                uint32_t v = src_row[i];
                dst_row[i] = (uint8_t)((v*new_maxval)/old_maxval);
            }
        } else if (old_bpc == 2 && new_bpc == 2) {
            /* 16 -> 16 */
            for (size_t i = 0; i < img_ptr->width*3; i++) {
                size_t o = i*2;
                uint16_t v = ((uint16_t)src_row[o] << 8) | (uint16_t)src_row[o+1];

                uint16_t r = (uint16_t)((v*new_maxval) / old_maxval);
                dst_row[o]      = (uint8_t)(r >> 8);
                dst_row[o+1]    = (uint8_t)(r);
            }
        } else if (old_bpc == 1 && new_bpc == 2) {
            /* 8 -> 16 */
            for (size_t i = 0; i < img_ptr->width*3; i++) {
                uint32_t v = src_row[i];

                size_t o = i*2;
                uint16_t r = (uint16_t)((v*new_maxval) / old_maxval);
                dst_row[o]      = (uint8_t)(r >> 8);
                dst_row[o+1]    = (uint8_t)(r);
            }
        } else {
            /* 16 -> 8 */
            for (size_t i = 0; i < img_ptr->width*3; i++) {
                size_t o = i*2;
                uint16_t v = ((uint16_t)src_row[o] << 8) | (uint16_t)src_row[o+1];

                dst_row[i] = (uint8_t)((v*new_maxval)/ old_maxval);
            }
        }
    }

    free(img_ptr->data);
    img_ptr->data = new_data;
    img_ptr->stride = new_stride;
    img_ptr->maxval = new_maxval;

    return 0;
}

int ppm_rgb_to_grayscale_scalar(PPM_ptr dst_ptr, const PPM_ptr src_ptr) {

    if (ppm_validate(src_ptr) < 0) {
        return -1;
    }

    if (dst_ptr->width != src_ptr->width || 
            dst_ptr->height != src_ptr->height || 
            dst_ptr->maxval != src_ptr->maxval || 
            dst_ptr->data_size != src_ptr->data_size) {
        return -2;
    }

    size_t bytes_per_channel = 1;
    if (src_ptr->maxval > 255)
        bytes_per_channel = 2;

    if (bytes_per_channel == 1) {
        for (size_t y = 0; y < src_ptr->height; ++y) {
            data_t src_row = src_ptr->data + y*src_ptr->stride;
            data_t dst_row = dst_ptr->data + y*dst_ptr->stride; 

            for (size_t i = 0; i < src_ptr->width; ++i) {
                
                size_t o = i*3;

                uint8_t R = src_row[o];
                uint8_t G = src_row[o+1];
                uint8_t B = src_row[o+2];

                // Calculate luminance
                uint8_t Y = (uint8_t)((299*R + 587*G + 114*B)/1000);
                dst_row[o] = Y;
                dst_row[o+1] = Y;
                dst_row[o+2] = Y;
            }

        }
    } else {
        for (size_t y = 0; y < src_ptr->height; ++y) {
            data_t src_row = src_ptr->data + y*src_ptr->stride;
            data_t dst_row = dst_ptr->data + y*dst_ptr->stride; 

            for (size_t i = 0; i < src_ptr->width; ++i) {
                
                size_t o = i*6;

                uint16_t R = (src_row[o] << 8) | src_row[o+1];
                uint16_t G = (src_row[o+2] << 8) | src_row[o+3];
                uint16_t B = (src_row[o+4] << 8) | src_row[o+5];

                // Calculate luminance
                uint16_t Y = (uint16_t)((299*R + 587*G + 114*B)/1000);
                dst_row[o] = (uint8_t)(Y >> 8);
                dst_row[o+1] = (uint8_t)(Y);
                
                dst_row[o+2] = (uint8_t)(Y >> 8);
                dst_row[o+3] = (uint8_t)(Y);            
                
                dst_row[o+4] = (uint8_t)(Y >> 8);
                dst_row[o+5] = (uint8_t)(Y);       
            }
        }

    }

    return 0;
}

int ppm_scale_scalar(PPM_ptr img_ptr, double scale, double bias) {
    
    if (ppm_validate(img_ptr) < 0)
        return -1;

    uint8_t bytes_per_channel = 1;
    if (img_ptr->maxval > 255)
        bytes_per_channel = 2;

    if (bytes_per_channel == 1) {
        for (size_t y = 0; y < img_ptr->height; ++y) {
            data_t row = img_ptr->data + y*img_ptr->stride;

            for (size_t i = 0; i < img_ptr->width*3; ++i) {
                row[i] = (uint8_t)(row[i]*scale + bias);
            }
        }
    } else {
        for (size_t y = 0; y < img_ptr->height; ++y) {
            data_t row = img_ptr->data + y*img_ptr->stride;

            for (size_t i = 0; i < img_ptr->width*6; ++i) {
                size_t o = i*2;
                uint16_t val = (row[o] << 8) | row[o+1];
                uint16_t new_val = (uint16_t)(val*scale + bias);
                row[o] = (uint8_t)(new_val >> 8);
                row[o+1] = (uint8_t)(new_val);
            }
        }
    }

    return 0;
}


