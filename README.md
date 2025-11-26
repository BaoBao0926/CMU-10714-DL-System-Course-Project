# CMU-10714-DL-System-Course-Project
This is the repository of course prohect of CMU course: 10714 Deep Learning System: Algotirhm and Implementation

# 0.Environment Setup

## 0.1 Clone Repo
```bash
# clone the repo
git clone https://github.com/BaoBao0926/CMU-10714-DL-System-Course-Project.git
```
## 0.2 Environment on Linux (Ubuntu/WSL)
You are free to build developer environment by both conda or uv, but you should make sure the environment is running on linux (either x86 Ubuntu systems or Windows WSL). 

If you are a window system user, please make sure your computer has WSL. You can refer to the guideline at here: https://www.bilibili.com/video/BV1tW42197za/?spm_id_from=333.1007.top_right_bar_window_default_collection.content.click



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

After you build the environment, please set global variable to(, which is optional):
```bash
%set_env PYTHONPATH ./needle
%set_env NEEDLE_BACKEND nd
```
To build the project, use the following commands in your terminal(especially you first clone this repo):
```bash
make reset
make
```

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
└── ops_fused.py         # defien the fused ops, which should be replaced into C++

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

**TODO**:
  - @shuaiwei, write C++ code for each fusd operator in needle/ops/ops_fused.py
 
## 3. Optimization on AMD GPU

**TODO**:
  - @shuaiwei, good luck to you


