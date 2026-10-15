/* mpi_hello.c - MPI start-up and correctness check for the Sol node check.
 * Times MPI_Init, checks MPI_Allreduce (sum of rank+1, max and min of rank) on every rank,
 * counts distinct hosts and averages 100 barriers. Rank 0 prints
 *   SOL_MPI_HELLO ranks=<n> hosts=<distinct> allreduce_ok=<0|1> init_s=<MPI_Init s on rank 0> barrier_us=<avg us>
 *
 * Build: mpicc -O2 -o mpi_hello mpi_hello.c
 * Usage: srun --mpi=pmix -n <ranks> ./mpi_hello
 */
#define _POSIX_C_SOURCE 200809L
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int cmp_names(const void *x, const void *y)
{
    return strcmp((const char *)x, (const char *)y);
}

int main(int argc, char **argv)
{
    struct timespec t0, t1;
    char name[MPI_MAX_PROCESSOR_NAME];
    char *names;
    long long mine, sum;
    int rank, size, len, rmax, rmin, ok, all_ok, hosts, i;
    double init_s, tb;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    MPI_Init(&argc, &argv);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    init_s = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    mine = rank + 1;
    MPI_Allreduce(&mine, &sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&rank, &rmax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&rank, &rmin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ok = sum == (long long)size * (size + 1) / 2 && rmax == size - 1 && rmin == 0;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    memset(name, 0, sizeof name);
    MPI_Get_processor_name(name, &len);
    names = malloc((size_t)size * MPI_MAX_PROCESSOR_NAME);
    if (!names) {
        fprintf(stderr, "mpi_hello: out of memory\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Allgather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                  MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    tb = MPI_Wtime();
    for (i = 0; i < 100; i++)
        MPI_Barrier(MPI_COMM_WORLD);
    tb = (MPI_Wtime() - tb) / 100.0 * 1e6;

    if (rank == 0) {
        qsort(names, (size_t)size, MPI_MAX_PROCESSOR_NAME, cmp_names);
        hosts = 0;
        for (i = 0; i < size; i++)
            if (i == 0 || strcmp(names + (size_t)i * MPI_MAX_PROCESSOR_NAME,
                                 names + (size_t)(i - 1) * MPI_MAX_PROCESSOR_NAME) != 0)
                hosts++;
        printf("SOL_MPI_HELLO ranks=%d hosts=%d allreduce_ok=%d init_s=%.4f barrier_us=%.2f\n",
               size, hosts, ok && all_ok, init_s, tb);
        fflush(stdout);
    }
    free(names);
    MPI_Finalize();
    return 0;
}
