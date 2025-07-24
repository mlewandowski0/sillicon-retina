from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='esim_torch',
    package_dir={'': 'src'},
    packages=['esim_torch'],
    ext_modules=[
        CUDAExtension(
            name='esim_torch.esim_cuda',
            sources=[
                'src/esim_torch/esim_cuda_kernel.cu',
            ]
        ),
        CUDAExtension(
            name='esim_torch.silicon_retina_cuda',
            sources=[
                'src/esim_torch/silicon_retina_cuda_kernel.cu',
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True)
    }
)
