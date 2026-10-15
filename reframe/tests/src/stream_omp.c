/* stream_omp.c - STREAM-style CPU memory bandwidth for the Sol node check.
 * Runs copy, scale, add and triad ntimes, keeps the best time of each (iteration 0 is not
 * counted), checks the final arrays against their closed-form values as STREAM does, and prints
 *   SOL_STREAM copy=<MB/s> scale=<MB/s> add=<MB/s> triad=<MB/s> threads=<n> elems=<N> valid=<0|1>
 *
 * Build: gcc -O3 -fopenmp -march=native -o stream_omp stream_omp.c   (without -fopenmp: 1 thread)
 * Usage: OMP_NUM_THREADS=<cores> OMP_PLACES=cores OMP_PROC_BIND=spread ./stream_omp N ntimes
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double now(void)
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
#endif
}

static double absd(double x)
{
    return x < 0 ? -x : x;
}

static double *alloc_array(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, 2 << 20, n * sizeof(double)) != 0) {
        fprintf(stderr, "stream_omp: cannot allocate %zu doubles\n", n);
        exit(1);
    }
    return p;
}

int main(int argc, char **argv)
{
    const double scalar = 3.0, eps = 1e-13;
    double best[4] = {1e30, 1e30, 1e30, 1e30}, t[4];
    double aj = 1.0, bj = 2.0, cj = 0.0, ea = 0.0, eb = 0.0, ec = 0.0;
    double *restrict a, *restrict b, *restrict c;
    size_t n, j;
    int ntimes, k, i, threads = 1, valid;
    char *end;

    if (argc != 3) {
        fprintf(stderr, "usage: %s N ntimes\n", argv[0]);
        return 1;
    }
    n = (size_t)strtoull(argv[1], &end, 10);
    ntimes = atoi(argv[2]);
    if (*end != '\0' || n < 1 || n > ((size_t)-1) / (3 * sizeof(double)) || ntimes < 2) {
        fprintf(stderr, "stream_omp: need N >= 1 and ntimes >= 2\n");
        return 1;
    }
    a = alloc_array(n);
    b = alloc_array(n);
    c = alloc_array(n);

#ifdef _OPENMP
#pragma omp parallel
#pragma omp single
    threads = omp_get_num_threads();
#endif

    /* first touch with the same static schedule as the kernels, so pages land on the right NUMA node */
#pragma omp parallel for schedule(static)
    for (j = 0; j < n; j++) {
        a[j] = 1.0;
        b[j] = 2.0;
        c[j] = 0.0;
    }

    for (k = 0; k < ntimes; k++) {
        t[0] = now();
#pragma omp parallel for schedule(static)
        for (j = 0; j < n; j++)
            c[j] = a[j];
        t[0] = now() - t[0];

        t[1] = now();
#pragma omp parallel for schedule(static)
        for (j = 0; j < n; j++)
            b[j] = scalar * c[j];
        t[1] = now() - t[1];

        t[2] = now();
#pragma omp parallel for schedule(static)
        for (j = 0; j < n; j++)
            c[j] = a[j] + b[j];
        t[2] = now() - t[2];

        t[3] = now();
#pragma omp parallel for schedule(static)
        for (j = 0; j < n; j++)
            a[j] = b[j] + scalar * c[j];
        t[3] = now() - t[3];

        for (i = 0; k > 0 && i < 4; i++)
            if (t[i] < best[i])
                best[i] = t[i];
    }
    for (i = 0; i < 4; i++)
        if (best[i] < 1e-9)
            best[i] = 1e-9;

    for (k = 0; k < ntimes; k++) {
        cj = aj;
        bj = scalar * cj;
        cj = aj + bj;
        aj = bj + scalar * cj;
    }
#pragma omp parallel for schedule(static) reduction(+:ea, eb, ec)
    for (j = 0; j < n; j++) {
        ea += absd(a[j] - aj);
        eb += absd(b[j] - bj);
        ec += absd(c[j] - cj);
    }
    valid = ea / (double)n / absd(aj) <= eps && eb / (double)n / absd(bj) <= eps &&
            ec / (double)n / absd(cj) <= eps;

    printf("SOL_STREAM copy=%.1f scale=%.1f add=%.1f triad=%.1f threads=%d elems=%zu valid=%d\n",
           1e-6 * 16.0 * (double)n / best[0], 1e-6 * 16.0 * (double)n / best[1],
           1e-6 * 24.0 * (double)n / best[2], 1e-6 * 24.0 * (double)n / best[3],
           threads, n, valid);
    free(a);
    free(b);
    free(c);
    return 0;
}
