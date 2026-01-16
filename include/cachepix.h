#pragma once
#include <stddef.h>

#define WHITESPACE_CHAR '\n'

typedef char* data_t;

typedef struct {
    uint32_t width, height, data_size;
    uint16_t maxval;
    char *data;       // aligned
    size_t stride;
} PPM_img, *PPM_ptr;

#define PIX_AT(i, x, y) i->data[y*i->stride + x*3]

/*
 * Load, Store, Clone, etc.
 */
PPM_ptr load_ppm_image(const char *file_name);
int save_ppm_image(PPM_ptr img_ptr, char *file_name, int force);
void free_ppm_image(PPM_ptr img_ptr);

PPM_ptr ppm_create(uint32_t width, uint32_t height, uint16_t maxval);
PPM_ptr ppm_clone(PPM_ptr src);
void ppm_clear(PPM_ptr img_ptr, uint16_t *val);

/*
 * Metadata access
 */
uint32_t ppm_width(const PPM_ptr img_ptr);
uint32_t ppm_height(const PPM_ptr img_ptr);
uint16_t ppm_maxval(const PPM_ptr img_ptr);
size_t ppm_stride(const PPM_ptr img_ptr);
data_t ppm_data(const PPM_ptr img_ptr);

/*
 * Validation functions
 */
int ppm_validate(const PPM_ptr img_ptr);
int ppm_validate_file(const char *file_name);
size_t ppm_expected_data_size(uint32_t width, uint32_t height, uint16_t maxval);

/*
 * Core PPM operations
 */

/*
 * Single pixel manipulation (not meant for loops because it's slow)
 */
int ppm_get_pixel(const PPM_ptr img_ptr, uint32_t x, uint32_t y, uint16_t *rgb);
int ppm_set_pixel(PPM_ptr img_ptr, uint32_t x, uint32_t y, const uint16_t *rgb);

// ####################### NOT IMPLEMENTED #######################
/*
 * Bulk operations (SIMD-friendly core)
 */
int ppm_copy(PPM_ptr dst_ptr, const PPM_ptr src_ptr);
int ppm_convert_maxval(PPM_ptr img_ptr, uint16_t new_maxval);
int ppm_rgb_to_grayscale(PPM_ptr dst_ptr, const PPM_ptr src_ptr);
int ppm_apply_scalar(PPM_ptr img_ptr, float scale, float bias);

/*
 * Memory layout and performance
 */
int ppm_realign(PPM_ptr img_ptr, size_t alignment);
int ppm_set_stride(PPM_ptr img_ptr, size_t stride);
int ppm_is_contiguous(const PPM_ptr img_ptr);

/*
 * CPU platform and features
 */
void ppm_init(void);
uint32_t ppm_cpu_features(void);

