#include "radonusfft.cuh"
#include "kernels.cu"
#include "shift.cu"

radonusfft::radonusfft(size_t ntheta, size_t pnz, size_t n, float center,
                       size_t theta_)
    : ntheta(ntheta), pnz(pnz), n(n), center(center) {
  float eps = 1e-3;
  mu = -log(eps) / (2 * n * n);
  m = ceil(2 * n * 1 / PI * sqrt(-mu * log(eps) + (mu * n) * (mu * n) / 4));
  cudaMalloc((void **)&f, n * n * pnz * sizeof(float2));
  cudaMalloc((void **)&g, n * ntheta * pnz * sizeof(float2));
  cudaMalloc((void **)&fde, 2 * n * 2 * n * pnz * sizeof(float2));
  cudaMalloc((void **)&fdee,
             (2 * n + 2 * m) * (2 * n + 2 * m) * pnz * sizeof(float2));

  cudaMalloc((void **)&x, n * ntheta * sizeof(float));
  cudaMalloc((void **)&y, n * ntheta * sizeof(float));
  cudaMalloc((void **)&theta, ntheta * sizeof(float));
  cudaMemcpy(theta, (float *)theta_, ntheta * sizeof(float), cudaMemcpyDefault);

  // Plan 2D FFTs
  int ffts[2] = {2 * n, 2 * n};
  int idist = 2 * n * 2 * n;
  int odist = (2 * n + 2 * m) * (2 * n + 2 * m);
  int inembed[2] = {2 * n, 2 * n};
  int onembed[2] = {2 * n + 2 * m, 2 * n + 2 * m};
  cufftPlanMany(&plan2dfwd, 2, ffts, inembed, 1, idist, onembed, 1, odist,
                CUFFT_C2C, pnz);
  cufftPlanMany(&plan2dadj, 2, ffts, onembed, 1, odist, inembed, 1, idist,
                CUFFT_C2C, pnz);

  // fft 1d
  ffts[0] = n;
  idist = n;
  odist = n;
  inembed[0] = n;
  onembed[0] = n;
  cufftPlanMany(&plan1d, 1, ffts, inembed, 1, idist, onembed, 1, odist,
                CUFFT_C2C, ntheta * pnz);
  cudaMalloc((void **)&shiftfwd, n * sizeof(float2));
  cudaMalloc((void **)&shiftadj, n * sizeof(float2));
  // compute shifts with respect to the rotation center
  takeshift <<<ceil(n / 1024.0), 1024>>> (shiftfwd, -(center - n / 2.0), n);
  takeshift <<<ceil(n / 1024.0), 1024>>> (shiftadj, (center - n / 2.0), n);

  BS2d = dim3(32, 32);
  BS3d = dim3(32, 32, 1);

  GS2d0 = dim3(ceil(n / (float)BS2d.x), ceil(ntheta / (float)BS2d.y));
  GS3d0 = dim3(ceil(n / (float)BS3d.x), ceil(n / (float)BS3d.y),
               ceil(pnz / (float)BS3d.z));
  GS3d1 = dim3(ceil(2 * n / (float)BS3d.x), ceil(2 * n / (float)BS3d.y),
               ceil(pnz / (float)BS3d.z));
  GS3d2 = dim3(ceil((2 * n + 2 * m) / (float)BS3d.x),
               ceil((2 * n + 2 * m) / (float)BS3d.y), ceil(pnz / (float)BS3d.z));
  GS3d3 = dim3(ceil(n / (float)BS3d.x), ceil(ntheta / (float)BS3d.y),
               ceil(pnz / (float)BS3d.z));
}

// destructor, memory deallocation
radonusfft::~radonusfft() { free(); }

void radonusfft::free() {
  if (!is_free) {
    cudaFree(f);
    cudaFree(g);
    cudaFree(fde);
    cudaFree(fdee);
    cudaFree(x);
    cudaFree(y);
    cudaFree(shiftfwd);
    cudaFree(shiftadj);
    cufftDestroy(plan2dfwd);
    cufftDestroy(plan2dadj);
    cufftDestroy(plan1d);
    is_free = true;
  }
}

void radonusfft::fwd(size_t g_, size_t f_) {
  cudaMemcpy(f, (float2 *)f_, n * n * pnz * sizeof(float2), cudaMemcpyDefault);
  cudaMemset(fde, 0, 2 * n * 2 * n * pnz * sizeof(float2));
  cudaMemset(fdee, 0, (2 * n + 2 * m) * (2 * n + 2 * m) * pnz * sizeof(float2));

  circ <<<GS3d0, BS3d>>> (f, 1.0f / n, n, pnz);
  takexy <<<GS2d0, BS2d>>> (x, y, theta, n, ntheta);

  divphi <<<GS3d0, BS3d>>> (fde, f, mu, n, pnz, TOMO_FWD);
  fftshiftc <<<GS3d1, BS3d>>> (fde, 2 * n, pnz);
  cufftExecC2C(plan2dfwd, (cufftComplex *)fde,
               (cufftComplex *)&fdee[m + m * (2 * n + 2 * m)], CUFFT_FORWARD);
  fftshiftc <<<GS3d2, BS3d>>> (fdee, 2 * n + 2 * m, pnz);

  wrap <<<GS3d2, BS3d>>> (fdee, n, pnz, m, TOMO_FWD);
  gather <<<GS3d3, BS3d>>> (g, fdee, x, y, m, mu, n, ntheta, pnz, TOMO_FWD);
  // shift with respect to given center
  shift <<<GS3d3, BS3d>>> (g, shiftfwd, n, ntheta, pnz);

  ifftshiftc <<<GS3d3, BS3d>>> (g, n, ntheta, pnz);
  cufftExecC2C(plan1d, (cufftComplex *)g, (cufftComplex *)g, CUFFT_INVERSE);
  ifftshiftc <<<GS3d3, BS3d>>> (g, n, ntheta, pnz);

  cudaMemcpy((float2 *)g_, g, n * ntheta * pnz * sizeof(float2),
             cudaMemcpyDefault);
}

void radonusfft::adj(size_t f_, size_t g_) {
  cudaMemcpy(g, (float2 *)g_, n * ntheta * pnz * sizeof(float2),
             cudaMemcpyDefault);
  cudaMemset(fde, 0, (2 * n + 2 * m) * (2 * n + 2 * m) * pnz * sizeof(float2));
  cudaMemset(fdee, 0, (2 * n + 2 * m) * (2 * n + 2 * m) * pnz * sizeof(float2));

  takexy <<<GS2d0, BS2d>>> (x, y, theta, n, ntheta);

  ifftshiftc <<<GS3d3, BS3d>>> (g, n, ntheta, pnz);
  cufftExecC2C(plan1d, (cufftComplex *)g, (cufftComplex *)g, CUFFT_FORWARD);
  ifftshiftc <<<GS3d3, BS3d>>> (g, n, ntheta, pnz);
  // applyfilter<<<GS3d3, BS3d>>>(g,n,ntheta,pnz);
  // shift with respect to given center
  shift <<<GS3d3, BS3d>>> (g, shiftadj, n, ntheta, pnz);

  gather <<<GS3d3, BS3d>>> (g, fdee, x, y, m, mu, n, ntheta, pnz, TOMO_ADJ);
  wrap <<<GS3d2, BS3d>>> (fdee, n, pnz, m, TOMO_ADJ);

  fftshiftc <<<GS3d2, BS3d>>> (fdee, 2 * n + 2 * m, pnz);
  cufftExecC2C(plan2dadj, (cufftComplex *)&fdee[m + m * (2 * n + 2 * m)],
               (cufftComplex *)fde, CUFFT_INVERSE);
  fftshiftc <<<GS3d1, BS3d>>> (fde, 2 * n, pnz);

  divphi <<<GS3d0, BS3d>>> (fde, f, mu, n, pnz, TOMO_ADJ);
  circ <<<GS3d0, BS3d>>> (f, 1.0f / n, n, pnz);

  cudaMemcpy((float2 *)f_, f, n * n * pnz * sizeof(float2),
             cudaMemcpyDefault);
}
