from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bev_pool_v2_ext',
    ext_modules=[
        CUDAExtension(
            'bev_pool_v2_ext',
            ['src/bev_pool.cpp', 'src/bev_pool_cuda.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']},
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
