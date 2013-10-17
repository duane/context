#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include <CoreServices/CoreServices.h>
#include <mach/mach_time.h>

// We include this for PAGE_SIZE
#include <mach/vm_param.h>

#define SIGMA_UNICODE "\xcf\x83"
#define SQUARED_UNICODE "\xc2\xb2"

const size_t CHUNK_SIZE = PAGE_SIZE;
const size_t MAX_MEM_SIZE = 0x80000000;
const size_t CHUNKS = MAX_MEM_SIZE / CHUNK_SIZE;
const size_t SAMPLES_PER_CHUNK = CHUNK_SIZE / sizeof(uint64_t);
const size_t MAX_SAMPLES = SAMPLES_PER_CHUNK * CHUNKS; // Our goal is 2GiG max.

typedef enum {
  NORMAL = 1,
  OUTLIER = 2,
} classification;

typedef struct {
  uint32_t sample;
  classification classification;
} classified_sample;

typedef struct {
   uint64_t **chunks;
   size_t numChunks;
   uint32_t entries;
} chunk_array;

chunk_array chunk_array_alloc(size_t entries, bool deep) {
  chunk_array array;
  array.entries = entries;
  array.numChunks = (entries / SAMPLES_PER_CHUNK) + 1;
  array.chunks = malloc(sizeof(uint64_t*) * array.numChunks);
  assert(array.chunks);
  if (deep) {
    for (size_t i = 0; i < array.numChunks; i++) {
      array.chunks[i] = malloc(CHUNK_SIZE);
      assert(array.chunks[i]);
    }
  }
  return array;
}

static inline uint64_t *entryInArrayAtIndex(chunk_array array, size_t index) {
  size_t chunkIndex = index / SAMPLES_PER_CHUNK;
  size_t chunkEntry = index % SAMPLES_PER_CHUNK;
  return &array.chunks[chunkIndex][chunkEntry];
}

uint64_t computeMean(chunk_array array) {
  double mean = 0.0;
  for (size_t n = 1; n <= array.entries; n++) {
    uint64_t x = *entryInArrayAtIndex(array, n - 1);
    double delta = (double)x - mean;
    mean = mean + (delta / n);
  }
  return mean;
}

void computeMeanAndStdDev(chunk_array array, double *mean, double *std) {
  assert(mean != NULL);
  assert(std != NULL);

  *mean = computeMean(array);
  *std = 0.0;
}

double gaussian(double x, double sigmaSquared, double mu) {
  double lhs = 1.0 / sqrt(2.0 * M_PI * sigmaSquared);
  double rhs = pow(M_E, -pow(x - mu, 2)/(2 * sigmaSquared));
  return lhs * rhs;
}

double probability(double x, double n, double sigmaSquared, double mu) {
  return gaussian(x, sigmaSquared, mu) * n;
}

classified_sample classifySample(uint64_t x, double n, double sigmaSquared, double mu) {
  classified_sample sample;
  sample.sample = (uint32_t) x;
  sample.classification = probability((double)x, n, sigmaSquared, mu) > 0.5 ? NORMAL : OUTLIER;
  return sample;
}

int main(int argc, char **argv) {
    // environment sanity check
    assert(sizeof(classified_sample) == sizeof(uint64_t));
    assert(__alignof__(classified_sample) == __alignof__(classified_sample));
    assert(sizeof(Nanoseconds) == sizeof(uint64_t));
    assert(__alignof__(UnsignedWide) <= __alignof__(uint64_t));
    assert(sizeof(AbsoluteTime) == sizeof(uint64_t));
    assert(__alignof__(AbsoluteTime) <= __alignof__(uint64_t));

    if (argc != 2) {
      fprintf(stderr, "USAGE: context [# of samples]\n");
      exit(0);
    }

    // Yes, this is bad code. atol is a horrible function. Nevertheless, my
    // time is better served than to write proper integer parsing code.
    size_t samples = atol(argv[1]);

    // semantic sanity check of input
    assert(samples > 1);
    assert(samples <= MAX_SAMPLES);

    chunk_array samples_array = chunk_array_alloc(samples, true);
    
    printf("Measuring context state...");
    fflush(stdout);
    
    for (size_t index = 0; index < samples; index++) {
      *(entryInArrayAtIndex(samples_array, index)) = mach_absolute_time();
    }

    printf("done.\n");

    printf("First pass (converting data and calculating mean and variance)...");
    fflush(stdout);
    // The first pass is designed to both convert the array from an array
    // nanoseconds since an arbitrary point to an array of intervals between
    // samples. Since we don't care about the original point in time, and
    // because we can calculate the interval with high precision, we
    // instead treat the intervals as the data itself.
    //
    // In addition, the first pass also calculates the mean and the variance incrementally.
    double mean = 0.0;
    double M2 = 0.0;
    size_t n;

    for (n = 1; n < samples_array.entries; n++) {
      uint64_t elapsed_absolute = *entryInArrayAtIndex(samples_array, n) - *entryInArrayAtIndex(samples_array, n-1);
      // convert from absolute units to nanoseconds
      // store the nanoseconds directly back into the array. We can do this
      // because we never access element n-1 again.
      uint64_t x;
      mach_timebase_info_data_t timebase;
      mach_timebase_info(&timebase);
      x = elapsed_absolute * (uint64_t)((double)timebase.numer / (double)timebase.denom);

    
      *entryInArrayAtIndex(samples_array, n - 1) = x;

      // now we update the mean for `x` and `n`.
      
      double delta = x - mean;
      mean = mean + delta / n;

      // and the variance estimate
      M2 = M2 + delta * (x - mean);
    }

    // now, we calculate variance and sample variance in terms of M2.
    double sigmaSquared = M2 / n;
    double sSquared = M2 / (n - 1);

    // and final standard deviation and sample standard deviation.
    double sigma = sqrt(sigmaSquared);
    double s = sqrt(sSquared);

    // because samples_array is now 1 shorter, we must check whether we need to
    // free the last chunk to avoid a leak.
    if (samples_array.entries % SAMPLES_PER_CHUNK == 1) { // the last sample was the only one in the last chunk
      samples_array.numChunks -= 1;
      free(samples_array.chunks[samples_array.numChunks]);
    }
    samples_array.entries -= 1;
    
    printf("done.\n");
    printf("Statistics:\n");
    printf("\tMean: %lf\n", mean);
    printf("\t" SIGMA_UNICODE SQUARED_UNICODE " = %lf\n", sigmaSquared);
    printf("\t" SIGMA_UNICODE " = %lf\n", sigma);
    printf("\tSample " SIGMA_UNICODE SQUARED_UNICODE " = %lf\n", sSquared);
    printf("\tSample " SIGMA_UNICODE " = %lf\n", s);

    /*printf("Second pass (computing normal & outlier statistics)...");
    
    double nMean = 0.0;
    double oMean = 0.0;
    double nM2 = 0.0;
    double oM2 = 0.0;

    for (n = 1; n < samples_array.entries; n++) {
      
    }*/

    printf("Dumping intervals to file...");
    fflush(stdout);
    FILE *file = fopen("./output.dat", "w");
    assert(file);

    // the disadvantage of using chunks is that we must also write the chunks
    // individually instead of asking stdio to write the entire array to a file
    // at once.

    // we could free the memory here, but the program is about to exit anyway,
    // so the unused memory will only last until the program exits.

    // write all but the last chunk. Then write only the number of entries in
    // the last chunk to file, which could be an entire chunk.
    size_t chunkIndex;
    for (chunkIndex = 0; chunkIndex < (samples_array.numChunks - 1); chunkIndex++) {
      int wrote = fwrite(samples_array.chunks[chunkIndex], sizeof(uint64_t), SAMPLES_PER_CHUNK, file);
      assert(wrote == SAMPLES_PER_CHUNK);
    }
    size_t entriesInLastChunk = samples_array.entries % SAMPLES_PER_CHUNK;
    assert(entriesInLastChunk == fwrite(samples_array.chunks[samples_array.numChunks - 1], sizeof(uint64_t), entriesInLastChunk, file));

    fclose(file);
    printf("done.\n");

    return 0;
}
