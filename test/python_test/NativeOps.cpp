/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#include "pytorch_npu_helper.hpp"

// A contiguous one-dimensional alias exposes the native bank's physical stride
// without ACLNN AutoContiguous materializing the entire persistent cache.
static at::Tensor bank_storage(const at::Tensor& bank, int64_t rank) {
  TORCH_CHECK(bank.dim() == rank && bank.size(0) > 0, "invalid native bank rank/slots");
  int64_t inner = 1;
  for (int64_t i = rank - 1; i > 0; --i) {
    TORCH_CHECK(bank.stride(i) == inner, "native bank inner axes must be contiguous");
    inner *= bank.size(i);
  }
  TORCH_CHECK(bank.stride(0) >= inner, "overlapping native slots are unsupported");
  return bank.as_strided({(bank.size(0) - 1) * bank.stride(0) + inner}, {1});
}

static at::Tensor mega_gdn_native_decode(
    const at::Tensor& qkv, const at::Tensor& z,
    const at::Tensor& b, const at::Tensor& a,
    const at::Tensor& weights, at::Tensor conv,
    const at::Tensor& a_log, const at::Tensor& dt_bias, at::Tensor ssm,
    const at::Tensor& ids, const at::Tensor& starts,
    const at::Tensor& accepted, const at::Tensor& norm) {
  TORCH_CHECK(qkv.dim() == 2 && z.dim() == 3 && z.size(2) == 128, "expected QKV[T,C], Z[T,Hv,128]");
  TORCH_CHECK(conv.dim() == 3 && ssm.dim() == 4 && conv.size(0) == ssm.size(0), "native bank shape mismatch");
  TORCH_CHECK(ssm.size(1) == z.size(1) && ssm.size(2) == 128 && ssm.size(3) == 128, "expected FP32 KV state");
  TORCH_CHECK(conv.size(2) == qkv.size(1), "convolution channel mismatch");
  TORCH_CHECK(ssm.scalar_type() == at::kFloat && conv.scalar_type() == at::kBFloat16, "state dtype mismatch");
  for (const auto& x : {z, b, a, weights, conv, a_log, dt_bias, ssm, ids, starts, accepted, norm})
    TORCH_CHECK(x.device() == qkv.device(), "all tensors must share the NPU device");
  auto qkv_flat = bank_storage(qkv, 2);
  auto z_flat = bank_storage(z, 3);
  auto conv_flat = bank_storage(conv, 3);
  auto ssm_flat = bank_storage(ssm, 4);
  auto convolved = at::empty(qkv.sizes(), qkv.options());
  // Padded output tokens not covered by a live metadata row remain zero.
  auto output = at::zeros(z.sizes(), z.options());
  int64_t conv_stride = conv.stride(0), ssm_stride = ssm.stride(0);
  int64_t conv_rows = conv.size(1), slots = conv.size(0);
  int64_t qkv_stride = qkv.stride(0), z_stride = z.stride(0);
  EXEC_NPU_CMD(aclnnMegaGdnNativeDecode,
      qkv_flat, z_flat, b, a, weights, conv_flat, a_log, dt_bias, ssm_flat,
      ids, starts, accepted, norm, conv_stride, ssm_stride,
      conv_rows, slots, qkv_stride, z_stride, convolved, conv_flat, ssm_flat, output);
  return output;
}

at::Tensor mega_gdn_native_prefill(
    const at::Tensor& mixed_qkv,
    const at::Tensor& b,
    const at::Tensor& a,
    const at::Tensor& z,
    const at::Tensor& conv_weight,
    at::Tensor& conv_state,
    const at::Tensor& a_log,
    const at::Tensor& dt_bias,
    const at::Tensor& conv_state_read_indices,
    const at::Tensor& conv_state_write_indices,
    const at::Tensor& ssm_state_read_indices,
    const at::Tensor& ssm_state_write_indices,
    at::Tensor& ssm_cache,
    const at::Tensor& mask_lower,
    const at::Tensor& mask_full,
    const at::Tensor& minus_identity,
    const at::Tensor& cu_seqlens,
    const at::Tensor& norm_weight,
    int64_t num_matrices) {
  uint64_t ffts_addr = 0;
  const char* soc_name = aclrtGetSocName();
  const bool is_ascend950 =
      soc_name != nullptr &&
      std::string(soc_name).find("Ascend950") != std::string::npos;
  // Ascend950 uses the operator's GM-based software synchronization and does
  // not consume an FFTS address. Keep the required ACLNN attribute at zero.
  // A2/A3 retain the hardware FFTS address path used by their kernel.
  if (!is_ascend950) {
    using RtGetC2cCtrlAddr = int32_t (*)(uint64_t*, uint32_t*);
    static RtGetC2cCtrlAddr get_c2c_ctrl_addr = [] {
      void* runtime = dlopen("libruntime.so", RTLD_LAZY | RTLD_LOCAL);
      TORCH_CHECK(runtime != nullptr,
                  "failed to load libruntime.so: ", dlerror());
      void* symbol = dlsym(runtime, "rtGetC2cCtrlAddr");
      TORCH_CHECK(symbol != nullptr,
                  "failed to resolve rtGetC2cCtrlAddr: ", dlerror());
      return reinterpret_cast<RtGetC2cCtrlAddr>(symbol);
    }();
    uint32_t ffts_len = 0;
    TORCH_CHECK(get_c2c_ctrl_addr(&ffts_addr, &ffts_len) == 0,
                "rtGetC2cCtrlAddr failed");
    TORCH_CHECK(ffts_len > 0,
                "rtGetC2cCtrlAddr returned an empty region");
  }
  TORCH_CHECK(num_matrices > 0, "num_matrices must be positive");
  int64_t ffts_addr_arg = static_cast<int64_t>(ffts_addr);

  auto conv_flat = bank_storage(conv_state, 3);
  auto ssm_flat = bank_storage(ssm_cache, 4);
  int64_t conv_stride = conv_state.stride(0), ssm_stride = ssm_cache.stride(0);
  int64_t conv_rows = conv_state.size(1), slots = conv_state.size(0);
  TORCH_CHECK(slots == ssm_cache.size(0), "native bank slot counts differ");
  at::Tensor out = at::empty_like(z);
  EXEC_NPU_CMD(aclnnMegaGdnNativePrefill,
               mixed_qkv,
               b,
               a,
               z,
               conv_weight,
               conv_flat,
               a_log,
               dt_bias,
               conv_state_read_indices,
               conv_state_write_indices,
               ssm_state_read_indices,
               ssm_state_write_indices,
               ssm_flat,
               mask_lower,
               mask_full,
               minus_identity,
               cu_seqlens,
               norm_weight,
               ffts_addr_arg,
               num_matrices, conv_stride, ssm_stride, conv_rows, slots,
               out,
               conv_flat,
               ssm_flat);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mega_gdn_native_prefill", &mega_gdn_native_prefill);
  m.def("mega_gdn_native_decode", &mega_gdn_native_decode,
        "Native KV state, ragged MTP and ordinary decode (910B)");
}
