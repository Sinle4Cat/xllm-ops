# Copyright 2026 The xLLM Authors. All Rights Reserved.

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE causal_conv1d_graph_a5_def.cpp)
endif()

add_ops_compile_options(
    OP_NAME CausalConv1dGraphA5
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../causal_conv1d/op_kernel
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE causal_conv1d_graph_a5 ACLNNTYPE aclnn)
endif()
