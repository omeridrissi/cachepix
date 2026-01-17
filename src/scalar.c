#include "cachepix.h"

int ppm_convert_maxval(PPM_ptr img_ptr, uint16_t new_maxval) {
    uint16_t old_maxval = img_ptr->maxval;
    
    uint32_t bytes_per_channel = 1;
    if (old_maxval > 255) 
        bytes_per_channel = 2;

    if (old_maxval <= 255 && new_maxval > 255) {
        return -1;  // Doesn't allow a change in maxval that affects the number of bytes per channel
    } else if (old_maxval >= 256 && new_maxval < 256) {
        return -1;
    } else {

        if (bytes_per_channel == 2) {
            for (int i = 0; i < img_ptr->data_size; i+=2) {
                double intensity = (*(uint16_t*)(img_ptr->data + i))/old_maxval;
                uint16_t new_val = (uint16_t)(intensity*new_maxval);
                img_ptr->data[i] = new_val;
            } 
        } else {
            for (int i = 0; i < img_ptr->data_size; i++) {
                double intensity = img_ptr->data[i]/old_maxval;
                uint16_t new_val = (uint16_t)(intensity*new_maxval);
                img_ptr->data[i] = new_val;
            }
        }        

        img_ptr->maxval = new_maxval;
    }
    return 0;
}



