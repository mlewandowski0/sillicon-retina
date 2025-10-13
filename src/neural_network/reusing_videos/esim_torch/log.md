(.venv) (base) mlew@mlew:~/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch$ pip install -e .
Obtaining file:///home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Preparing editable metadata (pyproject.toml) ... done
Building wheels for collected packages: esim_torch
  Building editable for esim_torch (pyproject.toml) ... error
  error: subprocess-exited-with-error
  
  × Building editable for esim_torch (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [626 lines of output]
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/_subclasses/functional_tensor.py:279: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:81.)
        cpu = _conversion_method_template(device=torch.device("cpu"))
      running editable_wheel
      creating /tmp/pip-wheel-dasxulgf/.tmp-urdelupl/esim_torch.egg-info
      writing /tmp/pip-wheel-dasxulgf/.tmp-urdelupl/esim_torch.egg-info/PKG-INFO
      writing dependency_links to /tmp/pip-wheel-dasxulgf/.tmp-urdelupl/esim_torch.egg-info/dependency_links.txt
      writing top-level names to /tmp/pip-wheel-dasxulgf/.tmp-urdelupl/esim_torch.egg-info/top_level.txt
      writing manifest file '/tmp/pip-wheel-dasxulgf/.tmp-urdelupl/esim_torch.egg-info/SOURCES.txt'
      reading manifest file '/tmp/pip-wheel-dasxulgf/.tmp-urdelupl/esim_torch.egg-info/SOURCES.txt'
      writing manifest file '/tmp/pip-wheel-dasxulgf/.tmp-urdelupl/esim_torch.egg-info/SOURCES.txt'
      creating '/tmp/pip-wheel-dasxulgf/.tmp-urdelupl/esim_torch-0.0.0.dist-info'
      creating /tmp/pip-wheel-dasxulgf/.tmp-urdelupl/esim_torch-0.0.0.dist-info/WHEEL
      running build_py
      running build_ext
      W1013 10:02:17.052000 11014 torch/utils/cpp_extension.py:517] There are no g++ version bounds defined for CUDA version 12.8
      building 'esim_torch.esim_cuda' extension
      creating /tmp/tmpdzdwicgh.build-temp/src/esim_torch
      W1013 10:02:17.052000 11014 torch/utils/cpp_extension.py:2425] TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation.
      W1013 10:02:17.052000 11014 torch/utils/cpp_extension.py:2425] If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'] to specific architectures.
      [1/1] /usr/local/cuda-12.8/bin/nvcc --generate-dependencies-with-compile --dependency-output /tmp/tmpdzdwicgh.build-temp/src/esim_torch/esim_cuda_kernel.o.d -I/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include -I/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.8/include -I/home/mlew/Desktop/PROJS/sillicon-retina/.venv/include -I/home/mlew/anaconda3/include/python3.13 -c -c /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/esim_cuda_kernel.cu -o /tmp/tmpdzdwicgh.build-temp/src/esim_torch/esim_cuda_kernel.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1018"' -DTORCH_EXTENSION_NAME=esim_cuda -gencode=arch=compute_120,code=compute_120 -gencode=arch=compute_120,code=sm_120 -std=c++17
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/esim_cuda_kernel.cu: In function ‘std::vector<at::Tensor> esim_forward_count_events(const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, float, float)’:
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/esim_cuda_kernel.cu:152:120: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        152 |   count_events_cuda_forward_kernel<float><<<blocks, threads>>>(
            |                                                                                                                        ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/esim_cuda_kernel.cu:152:147: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        152 |   count_events_cuda_forward_kernel<float><<<blocks, threads>>>(
            |                                                                                                                                                   ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/esim_cuda_kernel.cu:152:179: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        152 |   count_events_cuda_forward_kernel<float><<<blocks, threads>>>(
            |                                                                                                                                                                                   ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/esim_cuda_kernel.cu:152:211: warning: ‘T* at::Tensor::data() const [with T = long int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        152 |   count_events_cuda_forward_kernel<float><<<blocks, threads>>>(
            |                                                                                                                                                                                                                   ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/esim_cuda_kernel.cu: In function ‘at::Tensor esim_forward(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, float, float, int64_t)’:
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/esim_cuda_kernel.cu:199:112: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        199 |   esim_cuda_forward_kernel<float><<<blocks, threads>>>(
            |                                                                                                                ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/esim_cuda_kernel.cu:199:138: warning: ‘T* at::Tensor::data() const [with T = long int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        199 |   esim_cuda_forward_kernel<float><<<blocks, threads>>>(
            |                                                                                                                                          ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/esim_cuda_kernel.cu:199:165: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        199 |   esim_cuda_forward_kernel<float><<<blocks, threads>>>(
            |                                                                                                                                                                     ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/esim_cuda_kernel.cu:199:197: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        199 |   esim_cuda_forward_kernel<float><<<blocks, threads>>>(
            |                                                                                                                                                                                                     ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/esim_cuda_kernel.cu:199:228: warning: ‘T* at::Tensor::data() const [with T = long int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        199 |   esim_cuda_forward_kernel<float><<<blocks, threads>>>(
            |                                                                                                                                                                                                                                    ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/esim_cuda_kernel.cu:199:254: warning: ‘T* at::Tensor::data() const [with T = long int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        199 |   esim_cuda_forward_kernel<float><<<blocks, threads>>>(
            |                                                                                                                                                                                                                                                              ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/esim_cuda_kernel.cu:199:287: warning: ‘T* at::Tensor::data() const [with T = long int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        199 |   esim_cuda_forward_kernel<float><<<blocks, threads>>>(
            |                                                                                                                                                                                                                                                                                               ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      creating /tmp/tmpfbbek89l.build-lib/esim_torch
      g++ -pthread -B /home/mlew/anaconda3/compiler_compat -fno-strict-overflow -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/mlew/anaconda3/include -fPIC -O2 -isystem /home/mlew/anaconda3/include -shared -Wl,-rpath,/home/mlew/anaconda3/lib -Wl,-rpath-link,/home/mlew/anaconda3/lib -L/home/mlew/anaconda3/lib -Wl,-rpath,/home/mlew/anaconda3/lib -Wl,-rpath-link,/home/mlew/anaconda3/lib -L/home/mlew/anaconda3/lib /tmp/tmpdzdwicgh.build-temp/src/esim_torch/esim_cuda_kernel.o -L/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/lib -L/usr/local/cuda-12.8/lib64 -lc10 -ltorch -ltorch_cpu -ltorch_python -lcudart -lc10_cuda -ltorch_cuda -o /tmp/tmpfbbek89l.build-lib/esim_torch/esim_cuda.cpython-313-x86_64-linux-gnu.so
      building 'esim_torch.silicon_retina_cuda' extension
      W1013 10:02:42.637000 11014 torch/utils/cpp_extension.py:2425] TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation.
      W1013 10:02:42.637000 11014 torch/utils/cpp_extension.py:2425] If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'] to specific architectures.
      [1/2] c++ -MMD -MF /tmp/tmpdzdwicgh.build-temp/src/esim_torch/silicon_retina_cuda.o.d -pthread -B /home/mlew/anaconda3/compiler_compat -fno-strict-overflow -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/mlew/anaconda3/include -fPIC -O2 -isystem /home/mlew/anaconda3/include -fPIC -I/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include -I/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.8/include -I/home/mlew/Desktop/PROJS/sillicon-retina/.venv/include -I/home/mlew/anaconda3/include/python3.13 -c -c /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda.cpp -o /tmp/tmpdzdwicgh.build-temp/src/esim_torch/silicon_retina_cuda.o -O3 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1018"' -DTORCH_EXTENSION_NAME=silicon_retina_cuda -std=c++17
      FAILED: /tmp/tmpdzdwicgh.build-temp/src/esim_torch/silicon_retina_cuda.o
      c++ -MMD -MF /tmp/tmpdzdwicgh.build-temp/src/esim_torch/silicon_retina_cuda.o.d -pthread -B /home/mlew/anaconda3/compiler_compat -fno-strict-overflow -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/mlew/anaconda3/include -fPIC -O2 -isystem /home/mlew/anaconda3/include -fPIC -I/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include -I/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.8/include -I/home/mlew/Desktop/PROJS/sillicon-retina/.venv/include -I/home/mlew/anaconda3/include/python3.13 -c -c /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda.cpp -o /tmp/tmpdzdwicgh.build-temp/src/esim_torch/silicon_retina_cuda.o -O3 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1018"' -DTORCH_EXTENSION_NAME=silicon_retina_cuda -std=c++17
      In file included from /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/torch/csrc/Exceptions.h:12,
                       from /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/torch/csrc/api/include/torch/python.h:11,
                       from /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/torch/extension.h:9,
                       from /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda.cpp:2:
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h: In instantiation of ‘class pybind11::class_<HPF_MODE>’:
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h:2216:7:   required from ‘class pybind11::enum_<HPF_MODE>’
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda.cpp:91:38:   required from here
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h:1539:7: warning: ‘pybind11::class_<HPF_MODE>’ declared with greater visibility than its base ‘pybind11::detail::generic_type’ [-Wattributes]
       1539 | class class_ : public detail::generic_type {
            |       ^~~~~~
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h: In instantiation of ‘class pybind11::class_<BLUR_MODE>’:
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h:2216:7:   required from ‘class pybind11::enum_<BLUR_MODE>’
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda.cpp:97:40:   required from here
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h:1539:7: warning: ‘pybind11::class_<BLUR_MODE>’ declared with greater visibility than its base ‘pybind11::detail::generic_type’ [-Wattributes]
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h: In instantiation of ‘class pybind11::class_<SiliconRetinaParams>’:
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda.cpp:104:61:   required from here
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h:1539:7: warning: ‘pybind11::class_<SiliconRetinaParams>’ declared with greater visibility than its base ‘pybind11::detail::generic_type’ [-Wattributes]
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h: In instantiation of ‘pybind11::class_< <template-parameter-1-1>, <template-parameter-1-2> >::class_(pybind11::handle, const char*, const Extra& ...) [with Extra = {}; type_ = SiliconRetinaParams; options = {}]’:
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda.cpp:104:61:   required from here
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h:1599:28: warning: ‘pybind11::class_<SiliconRetinaParams>::class_<>(pybind11::handle, const char*)::<lambda(pybind11::detail::internals&)>’ declared with greater visibility than the type of its field ‘pybind11::class_<SiliconRetinaParams>::class_<>(pybind11::handle, const char*)::<lambda(pybind11::detail::internals&)>::<record capture>’ [-Wattributes]
       1599 |             with_internals([&](internals &internals) {
            |                            ^~~~~~~~~~~~~~~~~~~~~~~~~~~
       1600 |                 auto &instances = record.module_local ? get_local_internals().registered_types_cpp
            |                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       1601 |                                                       : internals.registered_types_cpp;
            |                                                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       1602 |                 instances[std::type_index(typeid(type_alias))]
            |                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       1603 |                     = instances[std::type_index(typeid(type))];
            |                     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       1604 |             });
            |             ~
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h: In instantiation of ‘pybind11::class_< <template-parameter-1-1>, <template-parameter-1-2> >::class_(pybind11::handle, const char*, const Extra& ...) [with Extra = {}; type_ = HPF_MODE; options = {}]’:
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h:2232:67:   required from ‘pybind11::enum_<Type>::enum_(const pybind11::handle&, const char*, const Extra& ...) [with Extra = {}; Type = HPF_MODE]’
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda.cpp:91:38:   required from here
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h:1599:28: warning: ‘pybind11::class_<HPF_MODE>::class_<>(pybind11::handle, const char*)::<lambda(pybind11::detail::internals&)>’ declared with greater visibility than the type of its field ‘pybind11::class_<HPF_MODE>::class_<>(pybind11::handle, const char*)::<lambda(pybind11::detail::internals&)>::<record capture>’ [-Wattributes]
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h: In instantiation of ‘pybind11::class_< <template-parameter-1-1>, <template-parameter-1-2> >::class_(pybind11::handle, const char*, const Extra& ...) [with Extra = {}; type_ = BLUR_MODE; options = {}]’:
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h:2232:67:   required from ‘pybind11::enum_<Type>::enum_(const pybind11::handle&, const char*, const Extra& ...) [with Extra = {}; Type = BLUR_MODE]’
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda.cpp:97:40:   required from here
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h:1599:28: warning: ‘pybind11::class_<BLUR_MODE>::class_<>(pybind11::handle, const char*)::<lambda(pybind11::detail::internals&)>’ declared with greater visibility than the type of its field ‘pybind11::class_<BLUR_MODE>::class_<>(pybind11::handle, const char*)::<lambda(pybind11::detail::internals&)>::<record capture>’ [-Wattributes]
      In file included from /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h:14:
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/detail/init.h: In instantiation of ‘Class* pybind11::detail::initimpl::construct_or_initialize(Args&& ...) [with Class = SiliconRetinaParams; Args = {int, int, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, HPF_MODE, float, HPF_MODE, float, bool}; typename std::enable_if<(! std::is_constructible<_Tp, _Args>::value), int>::type <anonymous> = 0]’:
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/detail/init.h:208:70:   required from ‘static void pybind11::detail::initimpl::constructor<Args>::execute(Class&, const Extra& ...) [with Class = pybind11::class_<SiliconRetinaParams>; Extra = {pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v}; typename std::enable_if<(! Class::has_alias), int>::type <anonymous> = 0; Args = {int, int, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, HPF_MODE, float, HPF_MODE, float, bool}]’
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h:1659:21:   required from ‘pybind11::class_< <template-parameter-1-1>, <template-parameter-1-2> >& pybind11::class_< <template-parameter-1-1>, <template-parameter-1-2> >::def(const pybind11::detail::initimpl::constructor<Args ...>&, const Extra& ...) [with Args = {int, int, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, HPF_MODE, float, HPF_MODE, float, bool}; Extra = {pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v}; type_ = SiliconRetinaParams; options = {}]’
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda.cpp:105:13:   required from here
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/detail/init.h:82:12: error: no matching function for call to ‘SiliconRetinaParams::SiliconRetinaParams(<brace-enclosed initializer list>)’
         82 |     return new Class{std::forward<Args>(args)...};
            |            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      In file included from /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda.cpp:5:
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/retina_params.h:72:5: note: candidate: ‘SiliconRetinaParams::SiliconRetinaParams(int, int, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, HPF_MODE, float, HPF_MODE, float, BLUR_MODE, BLUR_MODE, bool)’
         72 |     SiliconRetinaParams(int H = 256,
            |     ^~~~~~~~~~~~~~~~~~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/retina_params.h:97:35: note:   no known conversion for argument 24 from ‘bool’ to ‘BLUR_MODE’
         97 |                         BLUR_MODE blur1_mode = BLUR_MODE::BLUR_AVG,
            |                         ~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/retina_params.h:33:8: note: candidate: ‘constexpr SiliconRetinaParams::SiliconRetinaParams(const SiliconRetinaParams&)’
         33 | struct SiliconRetinaParams
            |        ^~~~~~~~~~~~~~~~~~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/retina_params.h:33:8: note:   candidate expects 1 argument, 24 provided
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/retina_params.h:33:8: note: candidate: ‘constexpr SiliconRetinaParams::SiliconRetinaParams(SiliconRetinaParams&&)’
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/retina_params.h:33:8: note:   candidate expects 1 argument, 24 provided
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h: In instantiation of ‘void pybind11::cpp_function::initialize(Func&&, Return (*)(Args ...), const Extra& ...) [with Func = pybind11::detail::initimpl::constructor<int, int, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, HPF_MODE, float, HPF_MODE, float, bool>::execute<pybind11::class_<SiliconRetinaParams>, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v>(pybind11::class_<SiliconRetinaParams>&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&)::<lambda(pybind11::detail::value_and_holder&, int, int, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, HPF_MODE, float, HPF_MODE, float, bool)>; Return = void; Args = {pybind11::detail::value_and_holder&, int, int, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, HPF_MODE, float, HPF_MODE, float, bool}; Extra = {pybind11::name, pybind11::is_method, pybind11::sibling, pybind11::detail::is_new_style_constructor, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v}]’:
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h:127:19:   required from ‘pybind11::cpp_function::cpp_function(Func&&, const Extra& ...) [with Func = pybind11::detail::initimpl::constructor<int, int, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, HPF_MODE, float, HPF_MODE, float, bool>::execute<pybind11::class_<SiliconRetinaParams>, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v>(pybind11::class_<SiliconRetinaParams>&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&)::<lambda(pybind11::detail::value_and_holder&, int, int, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, HPF_MODE, float, HPF_MODE, float, bool)>; Extra = {pybind11::name, pybind11::is_method, pybind11::sibling, pybind11::detail::is_new_style_constructor, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v}; <template-parameter-1-3> = void]’
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h:1621:22:   required from ‘pybind11::class_< <template-parameter-1-1>, <template-parameter-1-2> >& pybind11::class_< <template-parameter-1-1>, <template-parameter-1-2> >::def(const char*, Func&&, const Extra& ...) [with Func = pybind11::detail::initimpl::constructor<int, int, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, HPF_MODE, float, HPF_MODE, float, bool>::execute<pybind11::class_<SiliconRetinaParams>, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v>(pybind11::class_<SiliconRetinaParams>&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&, const pybind11::arg_v&)::<lambda(pybind11::detail::value_and_holder&, int, int, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, HPF_MODE, float, HPF_MODE, float, bool)>; Extra = {pybind11::detail::is_new_style_constructor, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v}; type_ = SiliconRetinaParams; options = {}]’
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/detail/init.h:205:15:   required from ‘static void pybind11::detail::initimpl::constructor<Args>::execute(Class&, const Extra& ...) [with Class = pybind11::class_<SiliconRetinaParams>; Extra = {pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v}; typename std::enable_if<(! Class::has_alias), int>::type <anonymous> = 0; Args = {int, int, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, HPF_MODE, float, HPF_MODE, float, bool}]’
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h:1659:21:   required from ‘pybind11::class_< <template-parameter-1-1>, <template-parameter-1-2> >& pybind11::class_< <template-parameter-1-1>, <template-parameter-1-2> >::def(const pybind11::detail::initimpl::constructor<Args ...>&, const Extra& ...) [with Args = {int, int, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, HPF_MODE, float, HPF_MODE, float, bool}; Extra = {pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v}; type_ = SiliconRetinaParams; options = {}]’
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda.cpp:105:13:   required from here
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h:244:40: error: static assertion failed: The number of argument annotations does not match the number of function arguments
        244 |             expected_num_args<Extra...>(
            |             ~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        245 |                 sizeof...(Args), cast_in::args_pos >= 0, cast_in::has_kwargs),
            |                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/pybind11/pybind11.h:244:40: note: ‘pybind11::detail::expected_num_args<pybind11::name, pybind11::is_method, pybind11::sibling, is_new_style_constructor, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v, pybind11::arg_v>(25, (((int)pybind11::detail::argument_loader<pybind11::detail::value_and_holder&, int, int, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, HPF_MODE, float, HPF_MODE, float, bool>::args_pos) >= 0), ((int)((bool)pybind11::detail::argument_loader<pybind11::detail::value_and_holder&, int, int, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, HPF_MODE, float, HPF_MODE, float, bool>::has_kwargs)))’ evaluates to false
      [2/2] /usr/local/cuda-12.8/bin/nvcc --generate-dependencies-with-compile --dependency-output /tmp/tmpdzdwicgh.build-temp/src/esim_torch/silicon_retina_cuda_kernel.o.d -I/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include -I/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.8/include -I/home/mlew/Desktop/PROJS/sillicon-retina/.venv/include -I/home/mlew/anaconda3/include/python3.13 -c -c /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu -o /tmp/tmpdzdwicgh.build-temp/src/esim_torch/silicon_retina_cuda_kernel.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -O3 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1018"' -DTORCH_EXTENSION_NAME=silicon_retina_cuda -gencode=arch=compute_120,code=compute_120 -gencode=arch=compute_120,code=sm_120 -std=c++17
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu: In function ‘std::vector<at::Tensor> forward_silicon_retina_verbose(const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, int, SiliconRetinaParams&)’:
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:523:48: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        523 |     run_silicon_retina_model(d_images.data<float>(),
            |                          ~~~~~~~~~~~~~~~~~~~~~~^~
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:523:72: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        523 |     run_silicon_retina_model(d_images.data<float>(),
            |                                                    ~                   ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:523:100: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        523 |     run_silicon_retina_model(d_images.data<float>(),
            |                                                                                                    ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:523:124: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        523 |     run_silicon_retina_model(d_images.data<float>(),
            |                                                                                                                            ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:523:152: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        523 |     run_silicon_retina_model(d_images.data<float>(),
            |                                                                                                                                                        ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:523:176: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        523 |     run_silicon_retina_model(d_images.data<float>(),
            |                                                                                                                                                                                ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:523:201: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        523 |     run_silicon_retina_model(d_images.data<float>(),
            |                                                                                                                                                                                                         ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:523:226: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        523 |     run_silicon_retina_model(d_images.data<float>(),
            |                                                                                                                                                                                                                                  ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:523:253: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        523 |     run_silicon_retina_model(d_images.data<float>(),
            |                                                                                                                                                                                                                                                             ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:523:282: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        523 |     run_silicon_retina_model(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                          ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:523:310: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        523 |     run_silicon_retina_model(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                      ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:523:334: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        523 |     run_silicon_retina_model(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                              ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:523:363: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        523 |     run_silicon_retina_model(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                                                           ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:523:393: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        523 |     run_silicon_retina_model(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                                                                                         ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:523:421: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        523 |     run_silicon_retina_model(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                                                                                                                     ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:523:450: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        523 |     run_silicon_retina_model(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:523:486: warning: ‘T* at::Tensor::data() const [with T = unsigned char]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        523 |     run_silicon_retina_model(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:523:523: warning: ‘T* at::Tensor::data() const [with T = unsigned char]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        523 |     run_silicon_retina_model(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu: In function ‘std::vector<at::Tensor> forward_silicon_retina(const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, int, SiliconRetinaParams&)’:
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:712:59: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        712 |     run_silicon_retina_model_fired_only(d_images.data<float>(),
            |                                     ~~~~~~~~~~~~~~~~~~~~~~^~
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:712:83: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        712 |     run_silicon_retina_model_fired_only(d_images.data<float>(),
            |                                                               ~                   ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:712:111: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        712 |     run_silicon_retina_model_fired_only(d_images.data<float>(),
            |                                                                                                               ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:712:135: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        712 |     run_silicon_retina_model_fired_only(d_images.data<float>(),
            |                                                                                                                                       ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:712:163: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        712 |     run_silicon_retina_model_fired_only(d_images.data<float>(),
            |                                                                                                                                                                   ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:712:187: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        712 |     run_silicon_retina_model_fired_only(d_images.data<float>(),
            |                                                                                                                                                                                           ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:712:212: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        712 |     run_silicon_retina_model_fired_only(d_images.data<float>(),
            |                                                                                                                                                                                                                    ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:712:237: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        712 |     run_silicon_retina_model_fired_only(d_images.data<float>(),
            |                                                                                                                                                                                                                                             ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:712:264: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        712 |     run_silicon_retina_model_fired_only(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                        ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:712:293: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        712 |     run_silicon_retina_model_fired_only(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                     ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:712:321: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        712 |     run_silicon_retina_model_fired_only(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                 ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:712:345: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        712 |     run_silicon_retina_model_fired_only(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                                         ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:712:374: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        712 |     run_silicon_retina_model_fired_only(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                                                                      ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:712:404: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        712 |     run_silicon_retina_model_fired_only(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                                                                                                    ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:712:432: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        712 |     run_silicon_retina_model_fired_only(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                                                                                                                                ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:712:461: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        712 |     run_silicon_retina_model_fired_only(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:712:497: warning: ‘T* at::Tensor::data() const [with T = unsigned char]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        712 |     run_silicon_retina_model_fired_only(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:712:534: warning: ‘T* at::Tensor::data() const [with T = unsigned char]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        712 |     run_silicon_retina_model_fired_only(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu: In function ‘std::vector<at::Tensor> forward_silicon_retina_online(const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, int, SiliconRetinaParams&)’:
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:893:55: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        893 |     run_silicon_retina_model_online(d_images.data<float>(),
            |                                 ~~~~~~~~~~~~~~~~~~~~~~^~
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:893:79: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        893 |     run_silicon_retina_model_online(d_images.data<float>(),
            |                                                           ~                   ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:893:103: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        893 |     run_silicon_retina_model_online(d_images.data<float>(),
            |                                                                                                       ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:893:131: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        893 |     run_silicon_retina_model_online(d_images.data<float>(),
            |                                                                                                                                   ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:893:159: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        893 |     run_silicon_retina_model_online(d_images.data<float>(),
            |                                                                                                                                                               ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:893:183: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        893 |     run_silicon_retina_model_online(d_images.data<float>(),
            |                                                                                                                                                                                       ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:893:208: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        893 |     run_silicon_retina_model_online(d_images.data<float>(),
            |                                                                                                                                                                                                                ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:893:233: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        893 |     run_silicon_retina_model_online(d_images.data<float>(),
            |                                                                                                                                                                                                                                         ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:893:260: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        893 |     run_silicon_retina_model_online(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                    ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:893:289: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        893 |     run_silicon_retina_model_online(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                 ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:893:317: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        893 |     run_silicon_retina_model_online(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                             ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:893:341: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        893 |     run_silicon_retina_model_online(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                                     ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:893:370: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        893 |     run_silicon_retina_model_online(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                                                                  ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:893:400: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        893 |     run_silicon_retina_model_online(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                                                                                                ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:893:428: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        893 |     run_silicon_retina_model_online(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                                                                                                                            ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:893:457: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        893 |     run_silicon_retina_model_online(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                         ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:893:493: warning: ‘T* at::Tensor::data() const [with T = unsigned char]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        893 |     run_silicon_retina_model_online(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      /home/mlew/Desktop/PROJS/sillicon-retina/src/neural_network/reusing_videos/esim_torch/src/esim_torch/silicon_retina_cuda_kernel.cu:893:530: warning: ‘T* at::Tensor::data() const [with T = unsigned char]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
        893 |     run_silicon_retina_model_online(d_images.data<float>(),
            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ^
      /tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
        247 |   T * data() const {
            | ^ ~~
      ninja: build stopped: subcommand failed.
      Traceback (most recent call last):
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/utils/cpp_extension.py", line 2595, in _run_ninja_build
          subprocess.run(
          ~~~~~~~~~~~~~~^
              command,
              ^^^^^^^^
          ...<4 lines>...
              check=True,
              ^^^^^^^^^^^
              env=env)
              ^^^^^^^^
        File "/home/mlew/anaconda3/lib/python3.13/subprocess.py", line 577, in run
          raise CalledProcessError(retcode, process.args,
                                   output=stdout, stderr=stderr)
      subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.
      
      The above exception was the direct cause of the following exception:
      
      Traceback (most recent call last):
        File "/home/mlew/Desktop/PROJS/sillicon-retina/.venv/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>
          main()
          ~~~~^^
        File "/home/mlew/Desktop/PROJS/sillicon-retina/.venv/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 373, in main
          json_out["return_val"] = hook(**hook_input["kwargs"])
                                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
        File "/home/mlew/Desktop/PROJS/sillicon-retina/.venv/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 303, in build_editable
          return hook(wheel_directory, config_settings, metadata_directory)
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/build_meta.py", line 468, in build_editable
          return self._build_with_temp_dir(
                 ~~~~~~~~~~~~~~~~~~~~~~~~~^
              cmd, ".whl", wheel_directory, config_settings
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          )
          ^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/build_meta.py", line 404, in _build_with_temp_dir
          self.run_setup()
          ~~~~~~~~~~~~~~^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/build_meta.py", line 317, in run_setup
          exec(code, locals())
          ~~~~^^^^^^^^^^^^^^^^
        File "<string>", line 4, in <module>
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/__init__.py", line 115, in setup
          return distutils.core.setup(**attrs)
                 ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/_distutils/core.py", line 186, in setup
          return run_commands(dist)
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/_distutils/core.py", line 202, in run_commands
          dist.run_commands()
          ~~~~~~~~~~~~~~~~~^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/_distutils/dist.py", line 1002, in run_commands
          self.run_command(cmd)
          ~~~~~~~~~~~~~~~~^^^^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/dist.py", line 1102, in run_command
          super().run_command(command)
          ~~~~~~~~~~~~~~~~~~~^^^^^^^^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
          cmd_obj.run()
          ~~~~~~~~~~~^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/command/editable_wheel.py", line 139, in run
          self._create_wheel_file(bdist_wheel)
          ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/command/editable_wheel.py", line 349, in _create_wheel_file
          files, mapping = self._run_build_commands(dist_name, unpacked, lib, tmp)
                           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/command/editable_wheel.py", line 272, in _run_build_commands
          self._run_build_subcommands()
          ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/command/editable_wheel.py", line 299, in _run_build_subcommands
          self.run_command(name)
          ~~~~~~~~~~~~~~~~^^^^^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/_distutils/cmd.py", line 357, in run_command
          self.distribution.run_command(command)
          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/dist.py", line 1102, in run_command
          super().run_command(command)
          ~~~~~~~~~~~~~~~~~~~^^^^^^^^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
          cmd_obj.run()
          ~~~~~~~~~~~^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/command/build_ext.py", line 96, in run
          _build_ext.run(self)
          ~~~~~~~~~~~~~~^^^^^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/_distutils/command/build_ext.py", line 368, in run
          self.build_extensions()
          ~~~~~~~~~~~~~~~~~~~~~^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/utils/cpp_extension.py", line 1072, in build_extensions
          build_ext.build_extensions(self)
          ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/_distutils/command/build_ext.py", line 484, in build_extensions
          self._build_extensions_serial()
          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/_distutils/command/build_ext.py", line 510, in _build_extensions_serial
          self.build_extension(ext)
          ~~~~~~~~~~~~~~~~~~~~^^^^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/command/build_ext.py", line 261, in build_extension
          _build_ext.build_extension(self, ext)
          ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/setuptools/_distutils/command/build_ext.py", line 565, in build_extension
          objects = self.compiler.compile(
              sources,
          ...<5 lines>...
              depends=ext.depends,
          )
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/utils/cpp_extension.py", line 856, in unix_wrap_ninja_compile
          _write_ninja_file_and_compile_objects(
          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
              sources=sources,
              ^^^^^^^^^^^^^^^^
          ...<11 lines>...
              with_cuda=with_cuda,
              ^^^^^^^^^^^^^^^^^^^^
              with_sycl=with_sycl)
              ^^^^^^^^^^^^^^^^^^^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/utils/cpp_extension.py", line 2227, in _write_ninja_file_and_compile_objects
          _run_ninja_build(
          ~~~~~~~~~~~~~~~~^
              build_directory,
              ^^^^^^^^^^^^^^^^
          ...<2 lines>...
              # that failed to build but there isn't a good way to get it here.
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              error_prefix='Error compiling objects for extension')
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/tmp/pip-build-env-2rxpj4z4/overlay/lib/python3.13/site-packages/torch/utils/cpp_extension.py", line 2612, in _run_ninja_build
          raise RuntimeError(message) from e
      RuntimeError: Error compiling objects for extension
      An error occurred when building editable wheel for esim_torch.
      See debugging tips in: https://setuptools.pypa.io/en/latest/userguide/development_mode.html#debugging-tips
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building editable for esim_torch
Failed to build esim_torch

[notice] A new release of pip is available: 25.1.1 -> 25.2
[notice] To update, run: pip install --upgrade pip
ERROR: Failed to build installable wheels for some pyproject.toml based projects (esim_torch)

