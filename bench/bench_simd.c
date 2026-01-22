#include "cachepix.h"
#include <stdio.h>
#include <time.h>
#include <stdint.h>

#define WARMUP_ITERATIONS 200
#define TEST_ITERATIONS 1000


int main() {
    struct timespec start, end;

    PPM_ptr src_ptr = ppm_load_image("resources/sample-image.ppm");
    PPM_ptr dst_ptr = ppm_create(src_ptr->width, src_ptr->height, src_ptr->maxval);
    
    printf("Running warm-up phase: getting CPU caches and clock speed ready...\n");

    // 1. Warm-up Phase: Get CPU caches and clock speed ready
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        ppm_rgb_to_grayscale_avx2(dst_ptr, src_ptr);        
    }

    printf("Finished %d iterations. Running timed phase...\n", WARMUP_ITERATIONS);

    // 2. Timed Phase
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < TEST_ITERATIONS; i++) {
        ppm_rgb_to_grayscale_avx2(dst_ptr, src_ptr);        
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    // 3. Calculation
    long seconds = end.tv_sec - start.tv_sec;
    long nanoseconds = end.tv_nsec - start.tv_nsec;
    double total_time_ns = (double)seconds * 1e9 + (double)nanoseconds;
    double average_time_ns = total_time_ns / TEST_ITERATIONS;

    printf("Total iterations: %d\n", TEST_ITERATIONS);
    printf("Average execution time: %.2f ms\n", average_time_ns / 1e6);

    ppm_free(src_ptr);
    ppm_free(dst_ptr);

    return 0;
}

