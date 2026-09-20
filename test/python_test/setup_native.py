from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension

root = Path(__file__).resolve().parent
setup(
    name="xllm-native-gdn",
    ext_modules=[
        NpuExtension(
            name="native_ops_lib",
            sources=[str(root / "NativeOps.cpp")],
            extra_compile_args=["-O2", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
