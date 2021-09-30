/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/layers/FfnLayer.h"

namespace fastertransformer {

template<typename T>
void FfnLayer<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
                          const std::vector<fastertransformer::Tensor>* input_tensors,
                          const FfnWeight<T>* ffn_weights)
{
    // input tensors:
    //      ffn_input [token_num, hidden_dimension],

    // output tensors:
    //      ffn_output [token_num, hidden_dimension],

    FT_CHECK(input_tensors->size() == 1);
    FT_CHECK(output_tensors->size() == 1);
    FT_CHECK(isValidTokenNum(input_tensors->at(0).shape[0]));
    allocateBuffer();

    const int m = input_tensors->at(0).shape[0];
    T* output_tensor = (T*)output_tensors->at(0).data;
    const T* input_tensor = (const T*)input_tensors->at(0).data;

#ifdef SPARSITY_ENABLED
    int m_tmp = input_tensors->at(0).shape[0];
    if (m_tmp % 8 != 0) {
        m_tmp = (m_tmp / 8 + 1) * 8;
    }
    const int m_padded = m_tmp;
    if (sparse_ && cublas_wrapper_->isUseSparse(1, inter_size_, m, hidden_units_)) {
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                inter_size_,
                                m_padded,
                                hidden_units_,
                                ffn_weights->intermediate_weight.sp_kernel,
                                input_tensor,
                                inter_buf_);
    } else {
#endif
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          inter_size_,
                          m,
                          hidden_units_,
                          ffn_weights->intermediate_weight.kernel,
                          inter_size_,
                          input_tensor,
                          hidden_units_,
                          inter_buf_,
                          inter_size_);
#ifdef SPARSITY_ENABLED
    }
#endif

    invokeAddBiasActivation(m, ffn_weights->intermediate_weight.bias);
    sync_check_cuda_error();

#ifdef SPARSITY_ENABLED
    if (sparse_ && cublas_wrapper_->isUseSparse(1, hidden_units_, m, inter_size_)) {
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                hidden_units_,
                                m_padded,
                                inter_size_,
                                ffn_weights->output_weight.sp_kernel,
                                inter_buf_,
                                output_tensor);
    } else {
#endif
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          hidden_units_,
                          m,
                          inter_size_,
                          ffn_weights->output_weight.kernel,
                          hidden_units_,
                          inter_buf_,
                          inter_size_,
                          output_tensor,
                          hidden_units_);
#ifdef SPARSITY_ENABLED
    }
#endif
    sync_check_cuda_error();
    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template<typename T>
FfnLayer<T>::FfnLayer(size_t max_batch_size,
                      size_t max_seq_len,
                      size_t head_num,
                      size_t size_per_head,
                      size_t inter_size,
                      cudaStream_t stream,
                      cublasMMWrapper* cublas_wrapper,
                      IAllocator* allocator,
                      bool is_free_buffer_after_forward,
                      bool sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_token_num_(max_batch_size * max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    inter_size_(inter_size),
    sparse_(sparse)
{
}

template<typename T>
FfnLayer<T>::FfnLayer(FfnLayer<T> const& ffn_layer):
    BaseLayer(
        ffn_layer.stream_, ffn_layer.cublas_wrapper_, ffn_layer.allocator_, ffn_layer.is_free_buffer_after_forward_),
    max_token_num_(ffn_layer.max_token_num_),
    head_num_(ffn_layer.head_num_),
    size_per_head_(ffn_layer.size_per_head_),
    hidden_units_(ffn_layer.hidden_units_),
    inter_size_(ffn_layer.inter_size_),
    sparse_(ffn_layer.sparse_)
{
}

template<typename T>
FfnLayer<T>::~FfnLayer()
{
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void FfnLayer<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        inter_buf_ = (T*)allocator_->malloc(sizeof(T) * max_token_num_ * inter_size_, false);
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void FfnLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(inter_buf_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool FfnLayer<T>::isValidTokenNum(size_t token_num)
{
    if (token_num <= max_token_num_) {
        return true;
    }
    else {
        freeBuffer();
        max_token_num_ = token_num * 1.2;
        return true;
    }
}

template class FfnLayer<float>;
template class FfnLayer<half>;

template<typename T>
GeluFfnLayer<T>::GeluFfnLayer(size_t max_batch_size,
                              size_t max_seq_len,
                              size_t head_num,
                              size_t size_per_head,
                              size_t inter_size,
                              cudaStream_t stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator* allocator,
                              bool is_free_buffer_after_forward,
                              bool sparse):
    FfnLayer<T>(max_batch_size,
                max_seq_len,
                head_num,
                size_per_head,
                inter_size,
                stream,
                cublas_wrapper,
                allocator,
                is_free_buffer_after_forward,
                sparse)
{
}

template<typename T>
GeluFfnLayer<T>::GeluFfnLayer(GeluFfnLayer<T> const& gelu_ffn_layer): FfnLayer<T>(gelu_ffn_layer)
{
}

template<typename T>
void GeluFfnLayer<T>::invokeAddBiasActivation(const int m, const T* bias)
{
    invokeAddBiasGelu<T>(inter_buf_, bias, m, inter_size_, stream_);
}

template class GeluFfnLayer<float>;
template class GeluFfnLayer<half>;

template<typename T>
ReluFfnLayer<T>::ReluFfnLayer(size_t max_batch_size,
                              size_t max_seq_len,
                              size_t head_num,
                              size_t size_per_head,
                              size_t inter_size,
                              cudaStream_t stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator* allocator,
                              bool is_free_buffer_after_forward,
                              bool sparse):
    FfnLayer<T>(max_batch_size,
                max_seq_len,
                head_num,
                size_per_head,
                inter_size,
                stream,
                cublas_wrapper,
                allocator,
                is_free_buffer_after_forward,
                sparse)
{
}

template<typename T>
ReluFfnLayer<T>::ReluFfnLayer(ReluFfnLayer<T> const& relu_ffn_layer): FfnLayer<T>(relu_ffn_layer)
{
}

template<typename T>
void ReluFfnLayer<T>::invokeAddBiasActivation(const int m, const T* bias)
{
    invokeAddBiasRelu<T>(inter_buf_, bias, m, inter_size_, stream_);
}

template class ReluFfnLayer<float>;
template class ReluFfnLayer<half>;

}  // namespace fastertransformer