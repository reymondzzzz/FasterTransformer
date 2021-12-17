#include "FasterTransformerPlugin.h"
#include "NvInferRuntimeCommon.h"

namespace fastertransformer {

FasterTransformerDecoderPlugin::FasterTransformerDecoderPlugin(
	int head_num,
	int size_per_head,
	int memory_hidden_units)
	: head_num(head_num), size_per_head(size_per_head), memory_hidden_units(memory_hidden_units) {
	check_cuda_error(cublasCreate(&cublas_handle_));
	check_cuda_error(cublasLtCreate(&cublas_lt_andle_));

	int device_id;
	check_cuda_error(cudaGetDevice(&device_id));
	allocator = std::make_shared<Allocator<AllocatorType::CUDA>>(device_id);

}

nvinfer1::IPluginV2DynamicExt *FasterTransformerDecoderPlugin::clone() const {

}

void FasterTransformerDecoderPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int32_t nbInputs,
																										 const nvinfer1::DynamicPluginTensorDesc *out, int32_t nbOutputs) {
	check_cuda_error(nbInputs == 1);
	check_cuda_error(nbOutputs == 1);
	int32_t batch_size = in[0].desc.dims.d[0];

	decoding_ = std::make_unique<DecodingSampling<OperationType::FP32>>(*allocator, batch_size, seq_len, head_num,
																																			size_per_head, vocab_size, decoder_layers,
																																			memory_hidden_units, memory_max_seq_len, start_id,
																																			end_id, candidate_num);

}

int32_t FasterTransformerDecoderPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
																								const nvinfer1::PluginTensorDesc *outputDesc,
																								const void *const *inputs,
																								void *const *outputs,
																								void *workspace,
																								cudaStream_t stream) {
	check_cuda_error(cublasSetStream(cublas_handle_, stream));
	decoding_->forward(decoder_params.data(), decoding_params);
}

}
