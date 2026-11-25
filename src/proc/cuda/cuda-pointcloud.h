// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 RealSense, Inc. All Rights Reserved.

#pragma once
#include "../pointcloud.h"
#include <memory>

// When CUDA is combined with NEON (ARM) or SSE (x86), inherit from the SIMD
// implementation to get its optimizations for methods not accelerated by CUDA
// (notably get_texture_map). Otherwise those fall back to the scalar code.
#if defined(__ARM_NEON) && defined(BUILD_WITH_NEON) && !defined(ANDROID)
#include "../neon/neon-pointcloud.h"
#elif defined(__SSSE3__)
#include "../sse/sse-pointcloud.h"
#endif

namespace rscuda { class pointcloud_cuda_helper; } // forward declaration (full def in cuda-pointcloud.cuh)

namespace librealsense
{
#if defined(__ARM_NEON) && defined(BUILD_WITH_NEON) && !defined(ANDROID)
    class pointcloud_cuda : public pointcloud_neon
#elif defined(__SSSE3__)
    class pointcloud_cuda : public pointcloud_sse
#else
    class pointcloud_cuda : public pointcloud
#endif
    {
    public:
        pointcloud_cuda();
    private:
        const float3 * depth_to_points(
            rs2::points output,
            const rs2_intrinsics &depth_intrinsics,
            const rs2::depth_frame& depth_frame) override;

        // Owns persistent device buffers; reused across frames. Held by shared_ptr
        // with a forward-declared type so this header stays free of CUDA includes.
        std::shared_ptr<rscuda::pointcloud_cuda_helper> _helper;
    };
}
