// run_inference.cpp
// Phase 2: load a serialized TensorRT engine and run one inference.
//
// All configuration is via the constants below — edit them and rebuild.
// setDLACore MUST happen on the runtime before deserialize if the engine was built for DLA.

#include "common.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// ---- Configuration (edit and rebuild) ---------------------------------------------------------
static const char* const kEnginePath  = "/mnt/workspace/cgz_workspace/Exercise/cuda_example/dla/output/model_dla0.trt";
static const int         kDlaCore     = 0;      // Must match the core the engine was built for.
static const int         kIterations  = 100;    // Warm-up is separate and excluded from timing.
static const char* const kInputPath   = "";     // "" -> feed zeros. Otherwise raw bytes concatenated
                                                // in the tensor order printed at load time.
static const char* const kOutputPrefix = "";    // "" -> only print byte-sum checksum.
                                                // Otherwise writes <prefix>.<tensor_name> per output.
static const char* const kInputName     = "input";
static const char* const kOutputname     = "output";                                            
// ------------------------------------------------------------------------------------------------

namespace {

struct Binding {
    std::string        name;
    nvinfer1::Dims     shape{};
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
    size_t             bytes = 0;
    void*              d_ptr = nullptr;
    bool               is_input = false;
};

}  // namespace

int main() {
    dla_example::TrtLogger logger;

    // 1) Deserialize engine ----------------------------------------------------------------------
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
        std::cerr << "createInferRuntime failed" << std::endl;
        return 1;
    }

    // 必须在deserializaeCudaEngine之前setDLACore
    // core可以设置成不同于build时指定的core，但是有可能会运行时失败。比如在build跟run时的core架构不适配
    const int nb_dla = runtime->getNbDLACores();
    if (nb_dla > 0 && kDlaCore >= 0 && kDlaCore < nb_dla) {
        runtime->setDLACore(kDlaCore);
        std::cout << "Runtime configured for DLA core " << kDlaCore
                  << " (platform has " << nb_dla << " cores)." << std::endl;
    } else if (nb_dla == 0) {
        std::cout << "Platform has no DLA cores; running on GPU." << std::endl;
    }

    std::vector<char> plan = dla_example::ReadBinaryFile(kEnginePath);
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (!engine) {
        std::cerr << "deserializeCudaEngine failed — engine file may be corrupt, or built for a"
                     " different TensorRT version / arch / DLA core." << std::endl;
        return 1;
    }

    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    if (!context) {
        std::cerr << "createExecutionContext failed" << std::endl;
        return 1;
    }

    // 2) 设置动态shape。需要使用engine中携带的额外profile进行设置。
    cudaStream_t stream_batch;
    DLA_EXAMPLE_CUDA_CHECK(cudaStreamCreate(&stream_batch));
    if(engine->getNbOptimizationProfiles() > 0) {
        std::cerr << "getNbOptimizationProfiles success " << std::endl;
        if(!context->setOptimizationProfileAsync(0, stream_batch)) {
            std::cerr << "setOptimizationProfileAsync failed" << std::endl;
            return 1;
        }

        nvinfer1::Dims intput_fixed_dim = engine->getProfileShape(kInputName, 0, nvinfer1::OptProfileSelector::kOPT);
        if(!context->setInputShape(kInputName, intput_fixed_dim)) {
            std::cerr << "setInputShape failed for " << kInputName << std::endl;
            return 1;
        }
        std::cerr << "fixed tsnsor " << kInputName << ": " << dla_example::DimsToString(intput_fixed_dim) << std::endl;

    } else {
        std::cerr << "getNbOptimizationProfiles failed " << std::endl;
    }

    // 3) Enumerate I/O tensors, allocate device buffers -----------------------------------------
    std::vector<Binding> bindings;
    const int nb_io = engine->getNbIOTensors();
    bindings.reserve(nb_io);

    for (int i = 0; i < nb_io; ++i) {
        Binding b;
        b.name  = engine->getIOTensorName(i);
        b.is_input =
            (engine->getTensorIOMode(b.name.c_str()) == nvinfer1::TensorIOMode::kINPUT);
        b.shape = engine->getTensorShape(b.name.c_str());
        b.dtype = engine->getTensorDataType(b.name.c_str());

        int64_t vol = dla_example::VolumeOf(b.shape);
        if (vol <= 0) {
            std::cerr << "Tensor " << b.name << " has non-positive volume; dynamic shapes not"
                         " handled by this demo. Re-export ONNX with fixed shapes." << std::endl;
            return 1;
        }
        b.bytes = static_cast<size_t>(vol) * dla_example::ElementSize(b.dtype);
        DLA_EXAMPLE_CUDA_CHECK(cudaMalloc(&b.d_ptr, b.bytes));
        context->setTensorAddress(b.name.c_str(), b.d_ptr);

        std::cout << "  " << (b.is_input ? "in " : "out") << " " << b.name
                  << " shape=" << dla_example::DimsToString(b.shape)
                  << " bytes=" << b.bytes << std::endl;
        bindings.push_back(b);
    }

    // 4) Populate inputs -------------------------------------------------------------------------
    const std::string input_path = kInputPath;
    if (!input_path.empty()) {
        std::vector<char> input_host = dla_example::ReadBinaryFile(input_path);
        size_t cursor = 0;
        for (auto& b : bindings) {
            if (!b.is_input) continue;
            if (cursor + b.bytes > input_host.size()) {
                std::cerr << "Input file too small: need " << b.bytes << " more bytes for "
                          << b.name << " but only " << (input_host.size() - cursor)
                          << " remain." << std::endl;
                return 1;
            }
            DLA_EXAMPLE_CUDA_CHECK(cudaMemcpy(b.d_ptr, input_host.data() + cursor, b.bytes,
                                              cudaMemcpyHostToDevice));
            cursor += b.bytes;
        }
    } else {
        for (auto& b : bindings) {
            if (b.is_input) {
                DLA_EXAMPLE_CUDA_CHECK(cudaMemset(b.d_ptr, 0, b.bytes));
            }
        }
        std::cout << "No input file configured; feeding zeros." << std::endl;
    }

    // 5) Execute ---------------------------------------------------------------------------------
    cudaStream_t stream;
    DLA_EXAMPLE_CUDA_CHECK(cudaStreamCreate(&stream));

    // Warm-up: first launch includes lazy CUDA/kernel init.
    if (!context->enqueueV3(stream)) {
        std::cerr << "enqueueV3 warm-up failed" << std::endl;
        return 1;
    }
    DLA_EXAMPLE_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < kIterations; ++it) {
        if (!context->enqueueV3(stream)) {
            std::cerr << "enqueueV3 failed at iteration " << it << std::endl;
            return 1;
        }
    }
    DLA_EXAMPLE_CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::steady_clock::now();
    double ms_total = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Ran " << kIterations << " iteration(s) in " << ms_total << " ms ("
              << (ms_total / kIterations) << " ms/iter, warm-up excluded)." << std::endl;

    // 6) Copy outputs and print checksums --------------------------------------------------------
    const std::string output_prefix = kOutputPrefix;
    for (auto& b : bindings) {
        if (b.is_input) continue;
        std::vector<char> host_out(b.bytes);
        DLA_EXAMPLE_CUDA_CHECK(cudaMemcpy(host_out.data(), b.d_ptr, b.bytes,
                                          cudaMemcpyDeviceToHost));

        uint64_t sum = 0;
        for (char c : host_out) {
            sum += static_cast<uint8_t>(c);
        }
        std::cout << "  output " << b.name << " byte-sum=" << sum << std::endl;

        if (!output_prefix.empty()) {
            dla_example::WriteBinaryFile(output_prefix + "." + b.name,
                                         host_out.data(), host_out.size());
        }
    }

    // 7) Cleanup ---------------------------------------------------------------------------------
    for (auto& b : bindings) {
        cudaFree(b.d_ptr);
    }
    cudaStreamDestroy(stream_batch);
    cudaStreamDestroy(stream);
    return 0;
}
