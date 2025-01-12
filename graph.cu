/**
 * Copyright (c) 2022, NVIDIA Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "gpu_graph.hpp"
#include "cuda_helper.hpp"
#include <iostream>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

constexpr int n_kernel = 10;
constexpr int n_iteration = 1;

__global__ void shortKernel(float *out_d, const float *in_d, int N, float f){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N) { 
      out_d[idx] = out_d[idx] + f + in_d[idx];
  }
}

__global__ void initKernel(float *ptr, int N, float f){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N) { 
      ptr[idx] = f;
  }
}

void run_kernels_graph(float *out_d, float *in_d, int size, float f, gpu_graph_t &g, cudaStream_t s)
{
  constexpr int threads = 256;
  int blocks = (size + threads - 1) / threads;

  for(int i = 0; i < n_kernel; i++){
    cudaKernelNodeParams params;
    params.blockDim = {static_cast<unsigned int>(threads), 1, 1};
    params.gridDim = {static_cast<unsigned int>(blocks), 1, 1};
    params.sharedMemBytes = 0;
    params.func = reinterpret_cast<void *>(shortKernel);
    void *args[] = {&out_d, &in_d, &size, &f};
    params.kernelParams = args;
    params.extra = nullptr;

    if (g.state() == gpu_graph_t::state_t::capture) {
      // Static kernels
      shortKernel<<<blocks, threads, 0, s>>>(out_d, in_d, size, 1.f);

      // // kernels with dynamic parameter `f`
      // // Add the kernel nodes
      g.add_kernel_node(i * 2 + 0, params, s);
    } else if (g.state() == gpu_graph_t::state_t::update) {
      // Update the kernel nodes
      g.update_kernel_node(i * 2 + 0, params);
    }
  } 
}

void run_kernels_no_graph(float *out_d, float *in_d, int size, float f, cudaStream_t s)
{
  constexpr int threads = 256;
  int blocks = (size + threads - 1) / threads;

  for(int i = 0; i < n_kernel; i++){
    // Static kernels
    shortKernel<<<blocks, threads, 0, s>>>(out_d, in_d, size, 1.f);

    // kernels with dynamic parameter `f`
    shortKernel<<<blocks, threads, 0, s>>>(out_d, in_d, size, f);
  } 
}

void run_init(float *ptr, int size, float f, cudaStream_t s) {
  constexpr int threads = 256;
  int blocks = (size + threads - 1) / threads;
  initKernel<<<blocks, threads, 0, s>>>(ptr, size, f);
}

void sync_and_print_output(float* out, float* out_d, int size) {
  cudaMemcpy(out, out_d, size * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < size; i++) {
    std::cout << out[i] << ",";
  }
  std::cout << "\n";
}

void fn(
    torch::Tensor& x) {
  x.sin_();
}

void stream_sync(
    at::cuda::CUDAStream& dependency,
    at::cuda::CUDAStream& dependent) {
  at::cuda::CUDAEvent cuda_ev;
  cuda_ev.record(dependency);
  cuda_ev.block(dependent);
}

int main() 
{
  gpu_graph_t _graph;
  gpu_graph_t _graph_always_recapture;

  _graph_always_recapture._always_recapture = true;

  torch::manual_seed(1);
  torch::cuda::manual_seed(1);
  torch::Device device(torch::kCUDA);
  auto x = torch::ones({3, 3}).to(device);

  auto output = torch::randn({3, 3}).to(device);
  std::cout << x << std::endl;

  auto captureStream = at::cuda::getStreamFromPool();
  auto stream = at::cuda::getCurrentCUDAStream();
  stream_sync(stream, captureStream);
  at::cuda::setCurrentCUDAStream(captureStream);

  printf("Running with    CUDA graph ('Recapture-then-update') ...\n");

  auto wrap_obj_no_graph = [&](gpu_graph_t &g, cudaStream_t s) {
    fn(x);
  };

  wrap_obj_no_graph(_graph_always_recapture, captureStream);

  std::cout << x << std::endl;

  x.fill_(1);

  _graph_always_recapture.wrap(wrap_obj_no_graph, captureStream);

  std::cout << x << std::endl;

  x = torch::zeros({3, 3}).to(device);
  x.fill_(0.5);
  _graph_always_recapture.wrap(wrap_obj_no_graph, captureStream);

  std::cout << x << std::endl;

  // Finalize memory, stream, events
  // cudaErrCheck(cudaStreamDestroy(stream));
  // cudaErrCheck(cudaEventDestroy(start));
  // cudaErrCheck(cudaEventDestroy(stop));

  // cudaErrCheck(cudaFree(out_d));
  // cudaErrCheck(cudaFree(in_d));
}
