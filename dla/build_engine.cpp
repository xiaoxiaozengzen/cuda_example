// build_engine.cpp
// Phase 1: parse an ONNX model and build a serialized TensorRT engine for DLA.
//
// All configuration is via the constants below — edit them and rebuild.

#include "common.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>

// ---- Configuration (edit and rebuild) ---------------------------------------------------------
static const char* const kOnnxPath      = "/mnt/workspace/cgz_workspace/Exercise/cuda_example/dla/model/normal.onnx";
static const char* const kEnginePath    = "/mnt/workspace/cgz_workspace/Exercise/cuda_example/dla/output/model_dla0.trt";
static const int         kDlaCore       = 0;      // Orin has cores 0 and 1.
static const bool        kUseInt8       = false;  // false -> FP16. INT8 needs QAT ranges in ONNX.
static const size_t      kWorkspaceMiB  = 512;
static const bool        kGpuFallback   = true;   // Layers not supported on DLA run on GPU.
static const char* const kInputName     = "input";
static const char* const kOutputname     = "output";

// ------------------------------------------------------------------------------------------------

int main() {
    dla_example::TrtLogger logger;

    // 1) Builder + network + parser --------------------------------------------------------------
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) {
        std::cerr << "createInferBuilder failed" << std::endl;
        return 1;
    }

    // Explicit batch is required for ONNX and for DLA.
    const uint32_t network_flags =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(network_flags));
    if (!network) {
        std::cerr << "createNetworkV2 failed" << std::endl;
        return 1;
    }

    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    const int parser_verbosity =
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);
    if (!parser->parseFromFile(kOnnxPath, parser_verbosity)) {
        std::cerr << "Failed to parse ONNX file: " << kOnnxPath << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "  " << parser->getError(i)->desc() << std::endl;
        }
        return 1;
    }
    std::cout << "Parsed ONNX: " << network->getNbInputs() << " input(s), "
              << network->getNbOutputs() << " output(s), "
              << network->getNbLayers() << " layer(s)." << std::endl;

    // 2) Builder config: DLA + precision ---------------------------------------------------------
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               kWorkspaceMiB * (1ULL << 20));

    // 2.1）remove dynamic shape
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    if(!profile) {
        std::cerr << "createOptimizationProfile failed" << std::endl;
        return 1;
    }
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 2;
    input_dims.d[0] = 10;
    input_dims.d[1] = 10;
    profile->setDimensions(kInputName, nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(kInputName, nvinfer1::OptProfileSelector::kOPT, input_dims);
    profile->setDimensions(kInputName, nvinfer1::OptProfileSelector::kMAX, input_dims);
    // 不要对输出张量设置其batch大小
    config->addOptimizationProfile(profile);

    const int32_t nb_dla_cores = builder->getNbDLACores();
    std::cout << "Platform reports " << nb_dla_cores << " DLA core(s)." << std::endl;
    const bool use_dla = (nb_dla_cores > 0) && (kDlaCore >= 0) && (kDlaCore < nb_dla_cores);

    if (use_dla) {
        // Layers that can't run on DLA either fail (no fallback) or are placed on GPU (fallback on).
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(kDlaCore);
        if (kGpuFallback) {
            config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        }
        std::cout << "Building for DLA core " << kDlaCore
                  << " (GPU fallback " << (kGpuFallback ? "on" : "off") << ")." << std::endl;
    } else if (nb_dla_cores == 0) {
        std::cout << "No DLA cores on this platform; building GPU-only engine." << std::endl;
    } else {
        std::cerr << "Requested DLA core " << kDlaCore
                  << " but only " << nb_dla_cores << " available; aborting." << std::endl;
        return 1;
    }

    // Precision. DLA requires FP16 or INT8.
    if (kUseInt8) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        // Also allow FP16 so any GPU-fallback layer that can't do INT8 uses FP16 instead of FP32.
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "INT8 mode without calibrator — assumes QAT (Q/DQ) ranges in the ONNX."
                  << std::endl;
    } else {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // 3) Build serialized engine -----------------------------------------------------------------
    std::cout << "Building engine (this can take several minutes for DLA)..." << std::endl;
    std::unique_ptr<nvinfer1::IHostMemory> plan(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
        std::cerr << "buildSerializedNetwork failed" << std::endl;
        return 1;
    }
    std::cout << "Engine built: " << plan->size() << " bytes." << std::endl;

    dla_example::WriteBinaryFile(kEnginePath, plan->data(), plan->size());
    std::cout << "Serialized engine to: " << kEnginePath << std::endl;
    return 0;
}
