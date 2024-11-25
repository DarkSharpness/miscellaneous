#include <atomic>
#include <coroutine>
#include <csignal>
#include <cstddef>
#include <cuda_runtime.h>
#include <exception>
#include <stdexcept>
#include <string>
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

struct promise;

struct my_coroutine : std::coroutine_handle<promise> {
  using promise_type = ::promise;
};

struct promise {
  auto get_return_object() -> my_coroutine {
    return {my_coroutine::from_promise(*this)};
  }
  auto initial_suspend() noexcept -> std::suspend_always { return {}; }
  auto final_suspend() noexcept -> std::suspend_always { return {}; }
  auto yield_value(auto x) { return x; }
  void return_void() {}
  void unhandled_exception() {}
};

static auto waste_gpu(int which) -> my_coroutine {
  constexpr auto kMax = std::size_t{1} << 40;
  constexpr auto kMin = std::size_t{1} << 20;
  auto current_size = std::size_t{1} << 30;
  cudaAssert(cudaSetDevice(which));
  while (running) {
    void *ptr = nullptr;
    bool success = true;
    try {
      cudaAssert(cudaMalloc(&ptr, current_size));
      current_size = current_size * 2;
    } catch (std::exception &e) {
      current_size = current_size / 2;
      success = false;
    }
    if (current_size > kMax)
      current_size = kMax;
    if (current_size < kMin)
      current_size = kMin;

    if (!success) {
      co_yield std::suspend_always{};
      cudaAssert(cudaSetDevice(which));
    }
  }
  co_return;
}

static auto stop(int) -> void { running = false; }

auto main() -> int {
  std::signal(SIGINT, stop);

  int device_count;
  cudaAssert(cudaGetDeviceCount(&device_count));
  std::vector<my_coroutine> coroutines;

  for (int i = 0; i < device_count; ++i)
    coroutines.push_back(waste_gpu(i));

  while (running)
    for (auto &coro : coroutines)
      if (coro)
        coro.resume();

  return 0;
}
