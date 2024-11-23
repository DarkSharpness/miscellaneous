#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cuda_runtime.h>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static std::atomic_bool running{true};

static void check_cuda(cudaError_t result, const char *, const char *file,
                       int line) {
  if (result != cudaSuccess) {
    std::string error = "CUDA Runtime Error: ";
    error += cudaGetErrorString(result);
    error += " at ";
    error += file;
    error += ":";
    error += std::to_string(line);
    throw std::runtime_error{error};
  }
}

#define cudaAssert(val) check_cuda((val), #val, __FILE__, __LINE__)

static auto waste_gpu() -> void {
  constexpr auto kMax = std::size_t{1} << 40;
  constexpr auto kMin = std::size_t{1} << 20;
  auto current_size = std::size_t{1} << 30;
  while (running) {
    if (current_size > kMax)
      current_size = kMax;
    if (current_size < kMin)
      current_size = kMin;
    auto ptr = (void *){};
    try {
      cudaAssert(cudaMalloc(&ptr, current_size));
      current_size *= 2;
    } catch (const std::exception &e) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      current_size /= 2;
    }
  }
}

static auto stop(int) -> void { running = false; }

auto main() -> int {
  int device_count;
  cudaAssert(cudaGetDeviceCount(&device_count));
  std::signal(SIGINT, stop);
  std::vector<std::thread> threads;
  for (auto i = 0; i < device_count; ++i) {
    cudaAssert(cudaSetDevice(i));
    threads.emplace_back(waste_gpu);
  }
  for (auto &thread : threads)
    thread.join();
  return 0;
}
