# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        mega_gdn_prefill_op_def.cpp
    )
endif()

set(MEGA_GDN_PREFILL_PROFILE_OPTIONS)
if(DEFINED ENV{MEGA_GDN_PREFILL_PROFILE_STOP})
    set(MEGA_GDN_PREFILL_PROFILE_STOP "$ENV{MEGA_GDN_PREFILL_PROFILE_STOP}")
    if(MEGA_GDN_PREFILL_PROFILE_STOP STREQUAL "conv")
        list(APPEND MEGA_GDN_PREFILL_PROFILE_OPTIONS -DE2E_STOP_AFTER_CONV)
    elseif(MEGA_GDN_PREFILL_PROFILE_STOP STREQUAL "frontend")
        list(APPEND MEGA_GDN_PREFILL_PROFILE_OPTIONS -DE2E_STOP_AFTER_FRONTEND)
    elseif(MEGA_GDN_PREFILL_PROFILE_STOP STREQUAL "cast")
        list(APPEND MEGA_GDN_PREFILL_PROFILE_OPTIONS -DE2E_STOP_AFTER_CAST)
    elseif(MEGA_GDN_PREFILL_PROFILE_STOP STREQUAL "cumsum")
        list(APPEND MEGA_GDN_PREFILL_PROFILE_OPTIONS -DMEGA_STOP_AFTER_CUMSUM)
    elseif(MEGA_GDN_PREFILL_PROFILE_STOP STREQUAL "transpose")
        list(APPEND MEGA_GDN_PREFILL_PROFILE_OPTIONS -DMEGA_STOP_AFTER_TRANSPOSE)
    elseif(MEGA_GDN_PREFILL_PROFILE_STOP STREQUAL "kkt")
        list(APPEND MEGA_GDN_PREFILL_PROFILE_OPTIONS -DMEGA_STOP_AFTER_KKT)
    elseif(MEGA_GDN_PREFILL_PROFILE_STOP STREQUAL "solve")
        list(APPEND MEGA_GDN_PREFILL_PROFILE_OPTIONS -DMEGA_STOP_AFTER_SOLVE)
    elseif(MEGA_GDN_PREFILL_PROFILE_STOP STREQUAL "wy")
        list(APPEND MEGA_GDN_PREFILL_PROFILE_OPTIONS -DMEGA_STOP_AFTER_WY)
    elseif(MEGA_GDN_PREFILL_PROFILE_STOP STREQUAL "h")
        list(APPEND MEGA_GDN_PREFILL_PROFILE_OPTIONS -DMEGA_STOP_AFTER_H)
    else()
        message(FATAL_ERROR
            "Unsupported MEGA_GDN_PREFILL_PROFILE_STOP=${MEGA_GDN_PREFILL_PROFILE_STOP}")
    endif()
endif()

add_ops_compile_options(
    OP_NAME MegaGdnPrefillOp
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Wno-error
            -DGDN_KERNEL_NAME=mega_gdn_prefill_op
            -DGDN_D=128
            -DGDN_C=128
            ${MEGA_GDN_PREFILL_PROFILE_OPTIONS}
            -I${CMAKE_CURRENT_SOURCE_DIR}
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../causal_conv1d/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../mega_chunk_gdn/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/pto-isa/include
            -I${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include
            -I${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc
            -I${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/profiling
            -I${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/runtime
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE mega_gdn_prefill_op ACLNNTYPE aclnn)
endif()
