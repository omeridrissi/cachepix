#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/stat.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>

#include "cachepix.h"


static int file_empty(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        return -1;      // file doesn't exist or error
    }
    return st.st_size == 0;
}

static int decimal_length_unsigned(uint32_t x)
{
    int len = 1;
    while (x >= 10) {
        x /= 10;
        ++len;
    }
    return len;
}

static int decimal_length_uint16(uint16_t x) {
    int len = 1;
    while (x >= 10) {
        x /= 10;
        ++len;
    }
    return len;
}

/*
 * Parse header function
 * Returns the header size if header is valid
 * Sets the values of width, height and maxval in the PPM_img structure
 * Returns a negative integer if header is invalid
 */
static int token_consume_header(PPM_ptr img_ptr, data_t file_buf, size_t file_size) {
    // Verify correct file signature
    if (!(file_buf[0] == 'P' && file_buf[1] == '6' && isspace(file_buf[2]))) {
        return -1;
    }

    int header_size = 0;
    char *end;
    for (size_t i = 2; i < file_size; i++) {
        if (isspace(file_buf[i])) {
            continue;
        }

        if (file_buf[i] == '#') {
            while (file_buf[i] != '\n') {
                i++;
            }
            continue;
        }

        if (isdigit(file_buf[i])) {
            if (img_ptr->width == 0) {
                img_ptr->width = (uint32_t)strtol(file_buf+i, &end, 10);
                i = end - file_buf;
                continue;
            }
            if (img_ptr->height == 0) {
                img_ptr->height = (uint32_t)strtol(file_buf+i, &end, 10);
                i = end - file_buf;
                continue;
            }
            if (img_ptr->maxval == 0) {
                img_ptr->maxval = (uint16_t)strtol(file_buf+i-1, &end, 10);
                header_size = end - file_buf;
                break;
            }
        }
        if (i == file_size-1) {
            header_size = -1;
            break;
        }
    }

    return header_size;
}

/*
 * Copy a valid PPM image data and metadata from disk into PPM structure
 * Discards any header comments
 */
PPM_ptr ppm_load_image(const char *file_name) {

    PPM_ptr img_ptr = ppm_create_empty();

    FILE *fp = fopen(file_name, "rb");
    if (fp == NULL) {
        fprintf(stderr, "%s: Could not open file.\n", file_name);
        return NULL;
    }

    fseek(fp, 0L, SEEK_END);
    size_t file_size = ftell(fp);
    fseek(fp, 0L, SEEK_SET);

    data_t entire_file = (data_t)malloc(sizeof(char)*file_size);
    fread(entire_file, sizeof(char), file_size, fp);

    fclose(fp);
    int header_size = token_consume_header(img_ptr, entire_file, file_size);
    
    if (header_size < 0) {
        fprintf(stderr, "%s: error parsing header: Could not read file format. Only PPM is supported.\n", file_name);
        return NULL;
    }

    size_t data_size = ppm_expected_data_size(img_ptr->width, img_ptr->height, img_ptr->maxval);
    data_t data = (data_t)malloc(sizeof(uint8_t)*data_size);

    int bytes_per_pixel = 3;
    if (img_ptr->maxval > 255)
        bytes_per_pixel = 6;

    size_t row_bytes = img_ptr->width*bytes_per_pixel;
    img_ptr->stride = (row_bytes + 63) & ~((size_t)63);

    for (int y = 0; y < img_ptr->height; ++y) {
        data_t dst_row = data + y * img_ptr->stride;
        data_t src_row = entire_file + header_size + y * row_bytes;

        memcpy(dst_row, src_row, row_bytes);
    }

    free(entire_file);

    img_ptr->data_size = data_size; 
    img_ptr->data = data;

    return img_ptr;
}

/*
 * Write the PPM_img structure from memory to disk as a PPM image file
 * Stops if the file it's writing to already exists, isn't empty and force option isn't enabled
 * Overwrites existing file only if force option is enabled
 */
int ppm_save_image(PPM_ptr img_ptr, char *file_name, int force) {

    if (!file_empty(file_name) && !force) {
        fprintf(stderr, "ERR: save_ppm_image: file already exists and is not empty. (toggle the force option to overwrite it)\n");
        return 0;
    }

    FILE *fp = fopen(file_name, "w");
    
    if (fp == NULL) {
        perror(file_name);
        return -1;
    }

    // Calculate ts (stackoverflow pull)
    int width_str_len = decimal_length_unsigned(img_ptr->width); //(int)((ceil(log10(img_ptr->width))+1)*sizeof(char));
    int height_str_len = decimal_length_unsigned(img_ptr->height); //(int)((ceil(log10(img_ptr->height))+1)*sizeof(char));
    int maxval_str_len = decimal_length_uint16(img_ptr->maxval); //(int)((ceil(log10(img_ptr->maxval))+1)*sizeof(char));

    char *width_str = (char*)malloc(sizeof(char)*width_str_len);
    char *height_str = (char*)malloc(sizeof(char)*height_str_len);
    char *maxval_str = (char*)malloc(sizeof(char)*maxval_str_len);

    sprintf(width_str, "%d", img_ptr->width);
    sprintf(height_str, "%d", img_ptr->height);
    sprintf(maxval_str, "%d", img_ptr->maxval);

    size_t file_size = ppm_expected_file_size(img_ptr->width, img_ptr->height, img_ptr->maxval);
    data_t entire_file = (data_t)malloc((file_size)*sizeof(char));

    // Construct fixed header
    entire_file[0] = 'P';
    entire_file[1] = '6';
    entire_file[2] = WHITESPACE_CHAR;
    int offset = 3;
    for (int i = 0; i < width_str_len; i++) {
        entire_file[i+offset] = width_str[i];
    }

    offset += width_str_len;
    entire_file[offset] = ' ';
    offset++;

    for (int i = 0; i < height_str_len; i++) {
        entire_file[i+offset] = height_str[i];
    }

    offset += height_str_len;
    entire_file[offset] = WHITESPACE_CHAR;
    offset++;

    for (int i = 0; i < maxval_str_len; i++) {
        entire_file[i+offset] = maxval_str[i];
    }
    offset += maxval_str_len;
    entire_file[offset] = WHITESPACE_CHAR;
    //offset++;
    
    // Finally copy the data

    // WRITEWITHSTRIDE
    int bytes_per_pixel = 3;
    if (img_ptr->maxval > 255)
        bytes_per_pixel = 6;

    size_t row_bytes = img_ptr->width*bytes_per_pixel;
    img_ptr->stride = (row_bytes + 63) & ~((size_t)63);

    for (int y = 0; y < img_ptr->height; ++y) {
        data_t src_row = img_ptr->data + y * img_ptr->stride;
        data_t dst_row = entire_file + offset + y * row_bytes;

        memcpy(dst_row, src_row, row_bytes);
    } 

    //memcpy(entire_file+offset, img_ptr->data, ppm_expected_data_size(img_ptr->width, img_ptr->height, img_ptr->maxval));

    size_t n_bytes = fwrite(entire_file, sizeof(char), file_size, fp);

    free(entire_file);
    fclose(fp);

    if (n_bytes < img_ptr->data_size) {
        fprintf(stderr, "ERROR: couldn't write all bytes to stream. %zu bytes written.\n", n_bytes);
        return -1;
    }

    printf("STAT: PPM image saved to %s successfully.\n", file_name);
    return 0;
}

void ppm_free(PPM_ptr img_ptr) {
    free(img_ptr->data);
    free(img_ptr);
}

PPM_ptr ppm_create(uint32_t width, uint32_t height, uint16_t maxval) {

    if (width == 0 || height == 0 || maxval == 0) {
        return NULL;
    }

    uint32_t bytes_per_channel = 1;
    if (maxval > 255)
        bytes_per_channel = 2;

    PPM_ptr img_ptr = (PPM_ptr)malloc(sizeof(PPM_img));
    data_t data = (data_t)malloc(sizeof(uint8_t)*ppm_expected_data_size(width, height, maxval));

    img_ptr->width = width;
    img_ptr->height = height;
    img_ptr->maxval = maxval;
    img_ptr->data_size = ppm_expected_data_size(width, height,maxval);
    img_ptr->data = data;
    
    size_t row_bytes = img_ptr->width*bytes_per_channel*3;
    img_ptr->stride = (row_bytes + 63) & ~((size_t)63);

    return img_ptr;
}

PPM_ptr ppm_create_empty() {

    PPM_ptr img_ptr = (PPM_ptr)malloc(sizeof(PPM_img));

    img_ptr->width = 0;
    img_ptr->height = 0;
    img_ptr->maxval = 0;
    img_ptr->data_size = 0;
    img_ptr->data = NULL;
    img_ptr->stride = 0;

    return img_ptr;
}

PPM_ptr ppm_clone(PPM_ptr src_ptr) {
    PPM_ptr dst_ptr = (PPM_ptr)malloc(sizeof(PPM_img));

    dst_ptr->width = src_ptr->width;
    dst_ptr->height = src_ptr->height;
    dst_ptr->maxval = src_ptr->maxval;
    dst_ptr->data_size = src_ptr->data_size;
    dst_ptr->data = src_ptr->data;
    dst_ptr->stride = src_ptr->stride;

    return dst_ptr;
}

void ppm_clear(PPM_ptr img_ptr, uint16_t *val) {
    for (size_t i = 0; i < (img_ptr->data_size/3); i+=3) {
        img_ptr->data[i] = val[0];
        img_ptr->data[i+1] = val[1];
        img_ptr->data[i+2] = val[2];
    }
}

/*
 * Metadata access
 */
uint32_t ppm_width(const PPM_ptr img_ptr) {
    return img_ptr->width;
}

uint32_t ppm_height(const PPM_ptr img_ptr) {
    return img_ptr->height;
}

uint16_t ppm_maxval(const PPM_ptr img_ptr) {
    return img_ptr->maxval;
}

size_t ppm_stride(const PPM_ptr img_ptr) {
    return img_ptr->stride;
}

data_t ppm_data(const PPM_ptr img_ptr) {
    return img_ptr->data;
}

/*
 * Validation functions
 */
int ppm_validate(const PPM_ptr img_ptr) {
    if (img_ptr == NULL || 
            img_ptr->data == NULL   || 
            img_ptr->width == 0     || 
            img_ptr->height == 0    || 
            img_ptr->maxval == 0    ||
            img_ptr->data_size != ppm_expected_data_size(img_ptr->width, img_ptr->height, img_ptr->maxval)) {
        return -1;
    }

    return 0;
}

size_t ppm_expected_data_size(uint32_t width, uint32_t height, uint16_t maxval) {
    size_t bytes_per_pixel = (maxval <= 255) ? 3 : 6;
    size_t row_bytes = width*bytes_per_pixel;
    size_t stride = (row_bytes + 63) & ~((size_t)63);

    return (size_t)(stride*height);
}

size_t ppm_expected_file_size(uint32_t width, uint32_t height, uint16_t maxval) {
    int width_str_len = decimal_length_unsigned(width); //(int)((ceil(log10(width))+1)*sizeof(char));
    int height_str_len = decimal_length_unsigned(height); //(int)((ceil(log10(height))+1)*sizeof(char));
    int maxval_str_len = decimal_length_uint16(maxval); //(int)((ceil(log10(maxval))+1)*sizeof(char));

    size_t bytes_per_pixel = (maxval <= 255) ? 3 : 6;

    return (size_t)(width*height*bytes_per_pixel + width_str_len + height_str_len + maxval_str_len + 7);

}

/*
 * Pixel manipulation 
 */

int ppm_get_pixel(const PPM_ptr img_ptr, uint32_t x, uint32_t y, uint16_t *rgb) {
    
    if (img_ptr == NULL || rgb == NULL) {
        return -1;
    }

    if (x >= img_ptr->width || y >= img_ptr->height) {
        return -1;
    }

    data_t pix_addr = &PIX_AT(img_ptr, x, y);
    rgb[0] = *pix_addr;
    rgb[1] = *(pix_addr+1);
    rgb[2] = *(pix_addr+2);

    return 0;
}

int ppm_set_pixel(PPM_ptr img_ptr, uint32_t x, uint32_t y, const uint16_t *rgb) {

    if (img_ptr == NULL || rgb == NULL) {
        return -1;
    }

    if (x >= img_ptr->width || y >= img_ptr->height) {
        return -1;
    }

    data_t pix_addr = &PIX_AT(img_ptr, x, y);
    *pix_addr = rgb[0];
    *(pix_addr+1) = rgb[1];
    *(pix_addr+2) = rgb[2];

    return 0;
}

/*
 * Bulk operations
 */
int ppm_copy(PPM_ptr dst_ptr, const PPM_ptr src_ptr) {

    if (ppm_validate(src_ptr) < 0) {
        return -1;
    }

    data_t dst_data = (data_t)malloc(src_ptr->data_size);

    dst_ptr->width = src_ptr->width;
    dst_ptr->height = src_ptr->height;
    dst_ptr->maxval = src_ptr->maxval;
    dst_ptr->data_size = src_ptr->data_size;
    dst_ptr->data = dst_data;
    dst_ptr->stride = src_ptr->stride;

    memcpy(dst_data, src_ptr->data, src_ptr->data_size);

    return 0;
}

/*
 * Alignment and performance
 */
int ppm_realign(PPM_ptr img_ptr, size_t alignment) {
    if (ppm_validate(img_ptr) < 0) {
        return -1;
    }

    // Alignment must be a power of two
    if ((alignment & (alignment - 1)) != 0)
        return -1;

    size_t bpp = (img_ptr->maxval <= 255) ? 3 : 6;
    size_t row_bytes = img_ptr->width * bpp;

    size_t new_stride = (row_bytes + alignment - 1) & ~(alignment - 1);

    if (img_ptr->stride == new_stride)
        return 0;

    data_t new_data = (data_t)malloc(new_stride + img_ptr->height);

    if (!new_data)
        return -1;

    for (size_t y = 0; y < img_ptr->height; y++) {
        data_t src_row = img_ptr->data + y * img_ptr->stride;
        data_t dst_row = new_data + y * new_stride;

        memcpy(dst_row, src_row, row_bytes);
    }

    free(img_ptr->data);
    img_ptr->data = new_data;
    img_ptr->stride = new_stride;
}
