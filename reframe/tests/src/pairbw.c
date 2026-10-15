/* pairbw.c - all-pairs bidirectional bandwidth and latency between nodes, one MPI rank per node.
 *
 * Round robin by the circle method: every pair of ranks is measured exactly once, and no rank
 * is in two pairs in the same round. In each round all pairs measure latency at the same time,
 * then take turns for bandwidth, one pair at a time, so no two exchanges share the fabric; a
 * pair keeps the best of BW_REPS runs. With the defaults, 107 nodes take about 3 minutes at
 * 200 Gb/s. On Linux each rank first binds to the CPUs next to the ib0 HCA, when sysfs shows it.
 * Rank 0 prints one SOL_PAIR line per pair as each round finishes, then one SOL_PAIRBW line.
 *
 * Usage: srun --mpi=pmix --ntasks-per-node=1 ./pairbw [msg_bytes [iters [lat_iters]]]
 *        defaults 16777216 20 1000
 * Build: mpicc -O2 -o pairbw pairbw.c
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE    /* sched_setaffinity and CPU_SET */
#endif
#define _POSIX_C_SOURCE 200112L

#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif

#define HOSTLEN 256
#define LAT_BYTES 8
#define LAT_WARMUP 10
#define BW_WARMUP 2
#define BW_REPS 2
#define TAG_LAT 11
#define TAG_BW 12

/* Partner of player me in round r of the circle method over m players (m even).
 * Player m-1 stays put: slot 0 pairs (m-1, r), slot k pairs (r+k, r-k) mod (m-1). */
static int partner(int me, int r, int m)
{
    int k = m - 1;

    if (me == k)
        return r;
    if (me == r)
        return k;
    return ((2 * r - me) % k + k) % k;
}

/* Bandwidth turn of player me in round r: its slot k (see partner), 0 .. m/2-1. Both players
 * of a pair get the same slot and no two pairs share one. */
static int slot_of(int me, int r, int m)
{
    int k = m - 1, d;

    if (me == k || me == r)
        return 0;
    d = ((me - r) % k + k) % k;
    return d < k - d ? d : k - d;
}

/* Run on the CPUs next to the HCA the site uses (ib0), from its sysfs local_cpulist such as
 * "0-23,48-71". Best effort: without the file, or if the kernel refuses, nothing changes. */
static void bind_near_hca(void)
{
#ifdef __linux__
    char list[4096], *s = list, *end;
    cpu_set_t set;
    long a, b;
    FILE *f = fopen("/sys/class/net/ib0/device/local_cpulist", "r");

    if (!f)
        return;
    if (!fgets(list, sizeof(list), f))
        list[0] = '\0';
    fclose(f);
    CPU_ZERO(&set);
    while (*s >= '0' && *s <= '9') {
        a = b = strtol(s, &end, 10);
        s = end;
        if (*s == '-') {
            b = strtol(s + 1, &end, 10);
            s = end;
        }
        for (; a <= b && a < CPU_SETSIZE; a++)
            CPU_SET(a, &set);
        if (*s == ',')
            s++;
    }
    if (CPU_COUNT(&set) > 0)
        (void)sched_setaffinity(0, sizeof(set), &set);
#endif
}

static long parse_arg(const char *s, long lo, long hi)
{
    char *end;
    long v = strtol(s, &end, 10);

    if (*s == '\0' || *end != '\0' || v < lo || v > hi)
        return -1;
    return v;
}

/* One-way latency in us: lat_iters ping-pongs of LAT_BYTES, the lower rank sends first. */
static double pingpong(int peer, int lower, int iters, char *buf)
{
    double t0 = 0.0, el;
    int i;

    for (i = 0; i < LAT_WARMUP + iters; i++) {
        if (i == LAT_WARMUP)
            t0 = MPI_Wtime();
        if (lower) {
            MPI_Send(buf, LAT_BYTES, MPI_BYTE, peer, TAG_LAT, MPI_COMM_WORLD);
            MPI_Recv(buf, LAT_BYTES, MPI_BYTE, peer, TAG_LAT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(buf, LAT_BYTES, MPI_BYTE, peer, TAG_LAT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(buf, LAT_BYTES, MPI_BYTE, peer, TAG_LAT, MPI_COMM_WORLD);
        }
    }
    el = MPI_Wtime() - t0;
    if (el <= 0.0)
        el = MPI_Wtick();
    return el / (2.0 * iters) * 1e6;
}

/* Bidirectional bandwidth in MB/s (1e6 B/s): both ranks send and receive bytes at once. */
static double exchange(int peer, char *sbuf, char *rbuf, int bytes, int iters)
{
    MPI_Request req[2];
    double t0 = 0.0, el;
    int i;

    for (i = 0; i < BW_WARMUP + iters; i++) {
        if (i == BW_WARMUP)
            t0 = MPI_Wtime();
        MPI_Irecv(rbuf, bytes, MPI_BYTE, peer, TAG_BW, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(sbuf, bytes, MPI_BYTE, peer, TAG_BW, MPI_COMM_WORLD, &req[1]);
        MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
    }
    el = MPI_Wtime() - t0;
    if (el <= 0.0)
        el = MPI_Wtick();
    return 2.0 * (double)bytes * iters / el / 1e6;
}

int main(int argc, char **argv)
{
    char host[HOSTLEN], latbuf[LAT_BYTES], *hosts = NULL, *sbuf, *rbuf, *dot;
    double res[3], *all = NULL;
    long msg_bytes = 16777216, iters = 20, lat_iters = 1000, pairs = 0;
    int rank, n, m, r, p, i, s, t;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n);

    if (argc > 1)
        msg_bytes = parse_arg(argv[1], 1, INT_MAX);
    if (argc > 2)
        iters = parse_arg(argv[2], 1, 1000000);
    if (argc > 3)
        lat_iters = parse_arg(argv[3], 1, 100000000);
    if (msg_bytes < 0 || iters < 0 || lat_iters < 0 || argc > 4) {
        if (rank == 0)
            fprintf(stderr, "usage: %s [msg_bytes [iters [lat_iters]]]\n", argv[0]);
        MPI_Finalize();
        return 2;
    }

    memset(host, 0, sizeof(host));
    if (gethostname(host, sizeof(host)) != 0)
        strcpy(host, "unknown");
    host[HOSTLEN - 1] = '\0';
    dot = strchr(host, '.');
    if (dot)
        *dot = '\0';

    bind_near_hca();
    sbuf = malloc((size_t)msg_bytes);
    rbuf = malloc((size_t)msg_bytes);
    if (rank == 0) {
        hosts = malloc((size_t)n * HOSTLEN);
        all = malloc((size_t)n * 3 * sizeof(double));
    }
    if (!sbuf || !rbuf || (rank == 0 && (!hosts || !all))) {
        fprintf(stderr, "pairbw: rank %d on %s: out of memory\n", rank, host);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    memset(sbuf, rank & 0xff, (size_t)msg_bytes);
    memset(rbuf, 0, (size_t)msg_bytes);
    memset(latbuf, 0, sizeof(latbuf));

    MPI_Gather(host, HOSTLEN, MPI_CHAR, hosts, HOSTLEN, MPI_CHAR, 0, MPI_COMM_WORLD);

    /* m players; with odd n, player n is a dummy and its partner sits the round out.
     * A barrier before each bandwidth turn keeps latency pairs clear of bulk traffic and lets
     * only one pair exchange at a time. */
    m = n + n % 2;
    for (r = 0; r < m - 1; r++) {
        double lat = 0.0, bw = 0.0;

        p = partner(rank, r, m);
        MPI_Barrier(MPI_COMM_WORLD);
        if (p < n)
            lat = pingpong(p, rank < p, (int)lat_iters, latbuf);
        for (s = 0; s < m / 2; s++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (p < n && slot_of(rank, r, m) == s)
                for (t = 0; t < BW_REPS; t++) {
                    double x = exchange(p, sbuf, rbuf, (int)msg_bytes, (int)iters);

                    if (x > bw)
                        bw = x;
                }
        }
        res[0] = (p < n && rank < p) ? p : -1.0;
        res[1] = bw;
        res[2] = lat;
        MPI_Gather(res, 3, MPI_DOUBLE, all, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            for (i = 0; i < n; i++) {
                int b = (int)all[3 * i];

                if (b < 0 || b >= n)
                    continue;
                printf("SOL_PAIR a=%s b=%s bibw_mbs=%.1f lat_us=%.2f\n",
                       hosts + (size_t)i * HOSTLEN, hosts + (size_t)b * HOSTLEN,
                       all[3 * i + 1], all[3 * i + 2]);
                pairs++;
            }
            fflush(stdout);
        }
    }

    if (rank == 0) {
        printf("SOL_PAIRBW nodes=%d pairs=%ld expected=%ld msg_bytes=%ld iters=%ld\n",
               n, pairs, (long)n * (n - 1) / 2, msg_bytes, iters);
        fflush(stdout);
    }

    free(sbuf);
    free(rbuf);
    free(hosts);
    free(all);
    MPI_Finalize();
    return 0;
}
