// gpu_check.cu - one-GPU check for the Sol node stage: device info, PCIe copy bandwidth,
// HBM triad bandwidth, and timed SGEMM/DGEMM whose every result is compared bit for bit
// with the first one. Uses device 0 of whatever CUDA_VISIBLE_DEVICES shows (run_gpu.sh
// sets it to one GPU-/MIG- UUID). Any CUDA or cuBLAS failure prints SOL_GPU_ERROR and exits 2.
//
// Build: nvcc -O3 -arch=native -o gpu_check gpu_check.cu -lcublas
// Usage: ./gpu_check --test info|pcie|hbm|gemm [--seconds S]

#ifndef _GNU_SOURCE
#define _GNU_SOURCE    // sched_setaffinity and CPU_SET (g++ defines it already)
#endif
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#ifdef __linux__
#include <sched.h>
#endif

static char uuid_str[96] = "unknown";
static cudaDeviceProp prop;

// Print SOL_GPU_ERROR and exit 2. "what" is cut at its first '(' so a stringified call
// reads as its function name; the reason stays one double-quoted token.
static void fail(const char *what, const char *why)
{
    char msg[320];
    snprintf(msg, sizeof msg, "%.*s: %s", (int)strcspn(what, "("), what, why ? why : "unknown error");
    for (char *p = msg; *p; p++)
        if (*p == '"' || *p == '\\' || *p == '[' || *p == ']' || *p == '\n' || *p == '\r')
            *p = ' ';
    printf("SOL_GPU_ERROR uuid=%s reason=\"%s\"\n", uuid_str, msg);
    fflush(stdout);
    exit(2);
}

static void cuda_ck(cudaError_t e, const char *what)
{
    if (e != cudaSuccess)
        fail(what, cudaGetErrorString(e));
}

static void cublas_ck(cublasStatus_t s, const char *what)
{
    if (s != CUBLAS_STATUS_SUCCESS) {
        char why[192];
        cudaError_t e = cudaGetLastError();
        snprintf(why, sizeof why, "%s%s%s", cublasGetStatusName(s),
                 e != cudaSuccess ? " / " : "", e != cudaSuccess ? cudaGetErrorString(e) : "");
        fail(what, why);
    }
}

#define CUDA_CK(call) cuda_ck((call), #call)
#define CUBLAS_CK(call) cublas_ck((call), #call)

static double now_s()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// A single GPU-/MIG- token in CUDA_VISIBLE_DEVICES is used verbatim, so the lines join with
// what run_gpu.sh passed; otherwise the UUID is formatted from cudaDeviceProp like nvidia-smi.
static void uuid_from_env()
{
    const char *v = getenv("CUDA_VISIBLE_DEVICES");
    if (!v || (strncmp(v, "GPU-", 4) != 0 && strncmp(v, "MIG-", 4) != 0) || strlen(v) >= sizeof uuid_str)
        return;
    for (const char *p = v; *p; p++)
        if (!isalnum((unsigned char)*p) && *p != '-' && *p != '/')
            return;
    snprintf(uuid_str, sizeof uuid_str, "%s", v);
}

static void uuid_from_prop()
{
    const unsigned char *b = (const unsigned char *)prop.uuid.bytes;
    snprintf(uuid_str, sizeof uuid_str,
             "%s-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             strstr(prop.name, "MIG") ? "MIG" : "GPU",
             b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
             b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
}

static void open_device()
{
    int count = 0;
    uuid_from_env();
    CUDA_CK(cudaGetDeviceCount(&count));
    if (count < 1)
        fail("cudaGetDeviceCount", "no CUDA device visible");
    CUDA_CK(cudaSetDevice(0));
    CUDA_CK(cudaGetDeviceProperties(&prop, 0));
    if (strcmp(uuid_str, "unknown") == 0)
        uuid_from_prop();
}

// Grid for the grid-stride kernels: enough 256-thread blocks to fill every SM.
static int grid_blocks()
{
    int per_sm = prop.maxThreadsPerMultiProcessor / 256;
    return prop.multiProcessorCount * (per_sm > 0 ? per_sm : 1);
}

__device__ __forceinline__ uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Deterministic values in [-1, 1) from a hash of (index, seed).
template <typename T>
__global__ void fill_kernel(T *p, size_t n, uint64_t seed)
{
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        uint64_t h = mix64(i ^ (seed << 56));
        p[i] = (T)((double)(h >> 11) * (1.0 / 4503599627370496.0) - 1.0);
    }
}

__global__ void triad_kernel(double *a, const double *b, const double *c, size_t n)
{
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        a[i] = b[i] + 3.0 * c[i];
}

// Wrapping uint64 sum of nw 64-bit words; launch with 256 threads per block.
__global__ void checksum_kernel(const unsigned long long *w, size_t nw, unsigned long long *out)
{
    __shared__ unsigned long long part[256];
    unsigned long long s = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < nw; i += stride)
        s += w[i];
    part[threadIdx.x] = s;
    __syncthreads();
    for (unsigned int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k)
            part[threadIdx.x] += part[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(out, part[0]);
}

static unsigned long long checksum(const void *buf, size_t bytes, unsigned long long *d_sum)
{
    unsigned long long s = 0;
    CUDA_CK(cudaMemset(d_sum, 0, sizeof s));
    checksum_kernel<<<grid_blocks(), 256>>>((const unsigned long long *)buf, bytes / 8, d_sum);
    CUDA_CK(cudaGetLastError());
    CUDA_CK(cudaMemcpy(&s, d_sum, sizeof s, cudaMemcpyDeviceToHost));
    return s;
}

static void test_info()
{
    char name[256];
    snprintf(name, sizeof name, "%s", prop.name);
    for (char *p = name; *p; p++)
        if (*p == '"' || *p == '\\')
            *p = ' ';
    printf("SOL_GPU_INFO uuid=%s name=\"%s\" mem_gib=%.1f sm=%d pci=%04x:%02x:%02x.0\n",
           uuid_str, name, (double)prop.totalGlobalMem / 1073741824.0, prop.multiProcessorCount,
           prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
}

// CPU numbers in a sysfs cpulist such as "0-23,48-71" -> cpus[]; returns how many were stored.
static int parse_cpulist(const char *s, int *cpus, int max)
{
    int n = 0;
    while (*s && *s != '\n') {
        char *end;
        long a = strtol(s, &end, 10), b;
        if (end == s || a < 0)
            break;
        b = a;
        s = end;
        if (*s == '-') {
            b = strtol(s + 1, &end, 10);
            if (end == s + 1)
                break;
            s = end;
        }
        for (long c = a; c <= b && n < max; c++)
            cpus[n++] = (int)c;
        if (*s == ',')
            s++;
        else
            break;
    }
    return n;
}

// Run on the CPUs next to the GPU (its sysfs local_cpulist) so the pinned PCIe buffer is
// NUMA-local and h2d/d2h do not depend on where the scheduler put us. Best effort.
static void bind_near_gpu()
{
    char path[96], list[4096];
    snprintf(path, sizeof path, "/sys/bus/pci/devices/%04x:%02x:%02x.0/local_cpulist",
             prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    FILE *f = fopen(path, "r");
    if (!f)
        return;
    char *got = fgets(list, sizeof list, f);
    fclose(f);
    static int cpus[4096];
    int n = got ? parse_cpulist(list, cpus, 4096) : 0;
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < n; i++)
        if (cpus[i] < CPU_SETSIZE)
            CPU_SET(cpus[i], &set);
    if (CPU_COUNT(&set) > 0)
        (void)sched_setaffinity(0, sizeof set, &set);
#else
    (void)n;
#endif
}

// Best GB/s of 10 timed copies after 2 warm-ups.
static double copy_gbs(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind,
                       cudaEvent_t e0, cudaEvent_t e1)
{
    double best = 0.0;
    for (int i = 0; i < 12; i++) {
        float ms = 0.0f;
        CUDA_CK(cudaEventRecord(e0, 0));
        CUDA_CK(cudaMemcpy(dst, src, bytes, kind));
        CUDA_CK(cudaEventRecord(e1, 0));
        CUDA_CK(cudaEventSynchronize(e1));
        CUDA_CK(cudaEventElapsedTime(&ms, e0, e1));
        if (i >= 2 && ms > 0.0f && (double)bytes / (ms * 1e-3) / 1e9 > best)
            best = (double)bytes / (ms * 1e-3) / 1e9;
    }
    return best;
}

static void test_pcie()
{
    const size_t bytes = (size_t)256 << 20;
    void *h = NULL, *d = NULL;
    cudaEvent_t e0, e1;
    bind_near_gpu();
    CUDA_CK(cudaMallocHost(&h, bytes));
    CUDA_CK(cudaMalloc(&d, bytes));
    memset(h, 0x5a, bytes);
    CUDA_CK(cudaMemset(d, 0, bytes));
    CUDA_CK(cudaEventCreate(&e0));
    CUDA_CK(cudaEventCreate(&e1));
    double h2d = copy_gbs(d, h, bytes, cudaMemcpyHostToDevice, e0, e1);
    double d2h = copy_gbs(h, d, bytes, cudaMemcpyDeviceToHost, e0, e1);
    printf("SOL_GPU_PCIE uuid=%s h2d_gbs=%.2f d2h_gbs=%.2f\n", uuid_str, h2d, d2h);
    CUDA_CK(cudaEventDestroy(e0));
    CUDA_CK(cudaEventDestroy(e1));
    CUDA_CK(cudaFree(d));
    CUDA_CK(cudaFreeHost(h));
}

static void test_hbm()
{
    size_t free_b = 0, total_b = 0;
    CUDA_CK(cudaMemGetInfo(&free_b, &total_b));
    size_t n = (size_t)1 << 28;
    size_t fit = (size_t)(0.6 * (double)free_b / 24.0);
    if (fit < n)
        n = fit;
    if (n < ((size_t)1 << 20))
        fail("hbm", "less than 24 MiB of free device memory");

    double *a = NULL, *b = NULL, *c = NULL;
    CUDA_CK(cudaMalloc(&a, n * sizeof(double)));
    CUDA_CK(cudaMalloc(&b, n * sizeof(double)));
    CUDA_CK(cudaMalloc(&c, n * sizeof(double)));
    int blocks = grid_blocks();
    fill_kernel<double><<<blocks, 256>>>(a, n, 1);
    fill_kernel<double><<<blocks, 256>>>(b, n, 2);
    fill_kernel<double><<<blocks, 256>>>(c, n, 3);
    CUDA_CK(cudaGetLastError());
    CUDA_CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CUDA_CK(cudaEventCreate(&e0));
    CUDA_CK(cudaEventCreate(&e1));
    double best = 0.0;
    for (int i = 0; i < 22; i++) {
        float ms = 0.0f;
        CUDA_CK(cudaEventRecord(e0, 0));
        triad_kernel<<<blocks, 256>>>(a, b, c, n);
        CUDA_CK(cudaGetLastError());
        CUDA_CK(cudaEventRecord(e1, 0));
        CUDA_CK(cudaEventSynchronize(e1));
        CUDA_CK(cudaEventElapsedTime(&ms, e0, e1));
        if (i >= 2 && ms > 0.0f && 24.0 * (double)n / (ms * 1e-3) / 1e9 > best)
            best = 24.0 * (double)n / (ms * 1e-3) / 1e9;
    }
    printf("SOL_GPU_HBM uuid=%s hbm_gbs=%.1f\n", uuid_str, best);
    CUDA_CK(cudaEventDestroy(e0));
    CUDA_CK(cudaEventDestroy(e1));
    CUDA_CK(cudaFree(a));
    CUDA_CK(cudaFree(b));
    CUDA_CK(cudaFree(c));
}

static void gemm(cublasHandle_t h, int n, const float *A, const float *B, float *C)
{
    const float one = 1.0f, zero = 0.0f;
    CUBLAS_CK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A, n, B, n, &zero, C, n));
}

static void gemm(cublasHandle_t h, int n, const double *A, const double *B, double *C)
{
    const double one = 1.0, zero = 0.0;
    CUBLAS_CK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A, n, B, n, &zero, C, n));
}

// GEMMs for `seconds`: the first result's checksum is the reference; C is poisoned before
// each later GEMM so one that writes nothing cannot match. Only the later GEMMs are timed.
template <typename T>
static double run_gemm(cublasHandle_t h, int n, T *A, T *B, T *C, double seconds,
                       unsigned long long *d_sum, long *iters, long *errors)
{
    size_t elems = (size_t)n * (size_t)n, bytes = elems * sizeof(T);
    fill_kernel<T><<<grid_blocks(), 256>>>(A, elems, 11);
    fill_kernel<T><<<grid_blocks(), 256>>>(B, elems, 12);
    CUDA_CK(cudaGetLastError());
    CUDA_CK(cudaMemset(C, 0xff, bytes));
    gemm(h, n, A, B, C);
    unsigned long long ref = checksum(C, bytes, d_sum);

    cudaEvent_t e0, e1;
    CUDA_CK(cudaEventCreate(&e0));
    CUDA_CK(cudaEventCreate(&e1));
    double ms_total = 0.0, t_end = now_s() + seconds;
    long it = 0;
    do {
        float ms = 0.0f;
        CUDA_CK(cudaMemset(C, 0xff, bytes));
        CUDA_CK(cudaEventRecord(e0, 0));
        gemm(h, n, A, B, C);
        CUDA_CK(cudaEventRecord(e1, 0));
        CUDA_CK(cudaEventSynchronize(e1));
        CUDA_CK(cudaEventElapsedTime(&ms, e0, e1));
        ms_total += ms;
        it++;
        if (checksum(C, bytes, d_sum) != ref)
            (*errors)++;
    } while (now_s() < t_end);
    CUDA_CK(cudaEventDestroy(e0));
    CUDA_CK(cudaEventDestroy(e1));
    *iters += it;
    return ms_total > 0.0 ? 2.0 * (double)n * (double)n * (double)n * (double)it / (ms_total * 1e-3) / 1e12 : 0.0;
}

static void test_gemm(double seconds)
{
    cublasHandle_t h;
    CUBLAS_CK(cublasCreate(&h));
    size_t free_b = 0, total_b = 0;
    CUDA_CK(cudaMemGetInfo(&free_b, &total_b));
    int n = 8192;
    while (n > 256 && 3.0 * (double)n * (double)n * 8.0 > 0.5 * (double)free_b)
        n /= 2;
    if (3.0 * (double)n * (double)n * 8.0 > 0.5 * (double)free_b)
        fail("gemm", "not enough free device memory");

    size_t bytes = (size_t)n * (size_t)n * sizeof(double);
    void *A = NULL, *B = NULL, *C = NULL;
    unsigned long long *d_sum = NULL;
    CUDA_CK(cudaMalloc(&A, bytes));
    CUDA_CK(cudaMalloc(&B, bytes));
    CUDA_CK(cudaMalloc(&C, bytes));
    CUDA_CK(cudaMalloc(&d_sum, sizeof *d_sum));

    long iters = 0, errors = 0;
    CUBLAS_CK(cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH));      // plain FP32, no TF32
    double s = run_gemm(h, n, (float *)A, (float *)B, (float *)C, seconds, d_sum, &iters, &errors);
    CUBLAS_CK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
    double d = run_gemm(h, n, (double *)A, (double *)B, (double *)C, seconds, d_sum, &iters, &errors);
    printf("SOL_GPU_GEMM uuid=%s sgemm_tflops=%.2f dgemm_tflops=%.2f iters=%ld errors=%ld\n",
           uuid_str, s, d, iters, errors);

    CUDA_CK(cudaFree(A));
    CUDA_CK(cudaFree(B));
    CUDA_CK(cudaFree(C));
    CUDA_CK(cudaFree(d_sum));
    CUBLAS_CK(cublasDestroy(h));
}

static int usage()
{
    fprintf(stderr, "usage: gpu_check --test info|pcie|hbm|gemm [--seconds S]\n");
    return 1;
}

int main(int argc, char **argv)
{
    const char *test = NULL;
    double seconds = 20.0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test") == 0 && i + 1 < argc)
            test = argv[++i];
        else if (strcmp(argv[i], "--seconds") == 0 && i + 1 < argc)
            seconds = atof(argv[++i]);
        else
            return usage();
    }
    if (!test || seconds <= 0.0 || (strcmp(test, "info") != 0 && strcmp(test, "pcie") != 0 &&
                                    strcmp(test, "hbm") != 0 && strcmp(test, "gemm") != 0))
        return usage();

    open_device();
    if (strcmp(test, "info") == 0)
        test_info();
    else if (strcmp(test, "pcie") == 0)
        test_pcie();
    else if (strcmp(test, "hbm") == 0)
        test_hbm();
    else
        test_gemm(seconds);
    CUDA_CK(cudaDeviceSynchronize());
    fflush(stdout);
    return 0;
}
