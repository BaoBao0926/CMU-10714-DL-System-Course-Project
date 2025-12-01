# CMU-10714-DL-System-Course-Project
This is the repository of course project of CMU course: 10714 Deep Learning System: Algotirhm and Implementation

# 0.Environment Setup

## 0.1 Clone Repo
```bash
# clone the repo
git clone https://github.com/BaoBao0926/CMU-10714-DL-System-Course-Project.git
```
## 0.2 Environment on Linux (Ubuntu/WSL)
You are free to build developer environment by both conda or uv, but you should make sure the environment is running on linux (either x86 Ubuntu systems or Windows WSL). 

If you are a window system user, please make sure your computer has WSL. You can refer to the [guideline](https://www.bilibili.com/video/BV1tW42197za/?spm_id_from=333.1007.top_right_bar_window_default_collection.content.click) at bilibili.



## 0.3 Environment Setup
You are free to build the environment by conda or uv

`Conda:`
```bash
# config the environment
conda create -n torch2needle python=3.10

# activate the environment
conda activate torch2needle

# install requirement
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

`uv: `
```bash
# config the environment
uv sync

# activate the environment
source /your-path-to-project/.venv/bin/activate

# you will see the environment in command line in:
(10714-hw4) Username@Host:your-path-to-project/
```
Our project maintains 3 branches that provides different features. Branch `main` is a test version for torch2needle only. The other two branch requires you to have an **AMD GPU** with **ROCm version >=7.1.0**. Please verify that before you run.

Branch `shuaiweh` add AMD GPU backend implementation library that enable the code to run on AMD GPU. Branch `profiling` adds profiling features on `shuaiweh`'s code to verify the performance of our implementation (such as inference time comparison).

If you have no AMD GPU, after the environment is built, you should set global variable to:
```bash
%set_env PYTHONPATH ./needle
%set_env NEEDLE_BACKEND nd
```
which will call operator defined on cpu or cuda.
To build the project, use the following commands in your terminal(especially you first clone this repo):
```bash
make reset
make
```

If you have an AMD GPU, after the environment is built, you should set global variable to:
```bash
%set_env PYTHONPATH ./needle
%set_env NEEDLE_BACKEND hip
```
which will call operators defined on AMD GPU.

To test if your environment is run correctly, you can run pytest command in hw4.ipynb for verification, for example:
```bash
!python3 -m pytest -l -v -k "language_model_implementation"
```
You will see all tests are passed just like this:
```bash
tests/hw4/test_sequence_models.py::test_language_model_implementation[cpu-rnn-1-True-1-1-1-1-1] ]9;4;1;0\PASSED
tests/hw4/test_sequence_models.py::test_language_model_implementation[cpu-rnn-1-True-1-1-1-1-13] ]9;4;1;0\PASSED
tests/hw4/test_sequence_models.py::test_language_model_implementation[cpu-rnn-1-True-1-1-1-2-1] ]9;4;1;0\PASSED
tests/hw4/test_sequence_models.py::test_language_model_implementation[cpu-rnn-1-True-1-1-1-2-13] ]9;4;1;0\PASSED
...
```

## 1.Torch2Needle

```text
torch2needle/
│
├── torch2needle_converter.py # define how to convert a PyTorch-based model into Needle-based model
├── weight_converter.py       # define how to load PyTorch weight into Needle model
├── utils.py                  # useful tools
└── torch_models.py           # torch-based models
```

If you want to add one more operator, you should add:

- (1) torch2needle/torch2needle_converter.py:
  - `convert_layer()`: you need add one more elif isinstance(layer, nn.*)
  - `convert_function_node()`: you need add one more elif op == operator.*
- (2) needle/nn/nn_basic.py:
  - you need add the code for new layers and operators
- (3) torch2needle/weight_converter.py:
  - `load_torch_weights_by_mapping()`: add the code to copy weight for each layer



## 2. Operaor Fusion

```text
operator_fusion/
│
├── operator_fusion.py   # main function of operator fusion.fuse_operators() is main function 
├── fusion_pattern.py    # define the pattern of each fused operation
└── fused_layer.py       # define the fused layer

needle/ops
└── ops_fused.py         # define the fused ops, which should be replaced into C++

```

If you want to add one more fused operator, you should add:

- (1) operator_fusion/operator_layer.py:
  - you need add fused operator here
- (2) needle/ops/ops_fused.py
  - add fused ope here
- (3) operator_fusion/fusion_pattern.py:
  - add new fusion pattern here
- (4) operator_fusion/operator_fusion.py
  - in starter function fuse_operator(), add new fused operator into `patterns = []`

If you have an AMD GPU and you have already set `NEEDLE_BACKEND` to `hip`, fused ops in ops_fused.py will automatically direct you to `ops_hip.py` that provides interface with AMD GPU backend. 
 
## 3. Optimization on AMD GPU

This AMD GPUs backend implementation is verified on AMD MI300X GPU, with **ROCm == 7.1.0** on **Ubuntu 24.04 LTS** system.

You can check the implementation at:
```text

needle/ops
└── ops_hip.py         # define AMD GPU (HIP) operator python interface

src
└── ndarray_backend_hip.cpp  # define specific implementation of AMD GPU (HIP) operators
```

Our implementation also provides a script to transfer cuda code to AMD GPU portable code (hip code). Before using this, please make sure your system has  `hipify-clang`, you can download them via [AMD ROCm introduction documentation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). Also, make sure your cuda code file is in `./src` directory.

Once you've checked your rocm and hipify-clang, please specify your `cuda-path` in `hipify-script.sh`, then, run `hipify-script.sh` in this command:

```bash
sh hipify-script.sh
```

if success, you'll find a hip version of your cuda code is generated in `src` directory. All you need to do is run `make` and make & cmake will compile everything for you！

## Profile our implementation

You are free to run `profile_full_pipeline_resnet.py` and `profile_full_pipelilne_unet.py` to profile our implementation after switch to `profiling` branch. Previous profile reports for these two models are also provided.