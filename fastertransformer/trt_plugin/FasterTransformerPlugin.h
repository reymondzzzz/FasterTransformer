//
// Created by kstarkov on 16.12.2021.
//

#ifndef FASTERTRANSFORMER_FASTERTRANSFORMERPLUGIN_H
#define FASTERTRANSFORMER_FASTERTRANSFORMERPLUGIN_H

#include <memory>
#include "NvInferPlugin.h"
#include "fastertransformer/utils/allocator.h"
#include "fastertransformer/decoding_sampling.h"

namespace fastertransformer {
//nvinfer1::Weights()

template<typename T>
class TransformerTrtTraits;

template<>
class TransformerTrtTraits<float> {
 public:
	static const OperationType OpType = OperationType::FP32;
	static const nvinfer1::DataType DataType = nvinfer1::DataType::kFLOAT;
};

template<>
class TransformerTrtTraits<half> {
 public:
	static const OperationType OpType = OperationType::FP16;
	static const nvinfer1::DataType DataType = nvinfer1::DataType::kHALF;
};

//template<typename T>
class FasterTransformerDecoderPlugin : public nvinfer1::IPluginV2DynamicExt {
 public:
	FasterTransformerDecoderPlugin(int head_num,
																 int size_per_head,
																 int memory_hidden_units);

	IPluginV2DynamicExt *clone() const _TENSORRT_OVERRIDE TRTNOEXCEPT;

	void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int32_t nbInputs,
											 const nvinfer1::DynamicPluginTensorDesc *out, int32_t nbOutputs) override;

	int32_t enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc,
									const void *const *inputs, void *const *outputs, void *workspace,
									cudaStream_t stream) override;

 private:
	std::unique_ptr<DecodingSampling<OperationType::FP32>> decoding_;
	// decoder params
	int seq_len = 12;
	int head_num = 2;
	int size_per_head = 64;
	int vocab_size = 42;
	int decoder_layers = 1;
	int memory_hidden_units = 256;
	int memory_max_seq_len = 288;
	int start_id = 1, end_id = 41;
	int candidate_num = 1;
	// decoder weights
	std::vector<DecoderInitParam<float>> decoder_params;
	DecodingInitParam<float> decoding_params;


	std::shared_ptr<fastertransformer::Allocator<AllocatorType::CUDA>> allocator;
	cublasHandle_t cublas_handle_;
	cublasLtHandle_t cublas_lt_andle_;
};

//    class ff : public nvinfer1::IPluginCreator

//    REGISTER_TENSORRT_PLUGIN(FasterTransformerDecoderPlugin)
}


#endif //FASTERTRANSFORMER_FASTERTRANSFORMERPLUGIN_H
