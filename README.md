# TinyExp

[![Release](https://img.shields.io/github/v/release/zengarden/TinyExp)](https://img.shields.io/github/v/release/zengarden/TinyExp)
[![Build status](https://img.shields.io/github/actions/workflow/status/zengarden/TinyExp/main.yml?branch=main)](https://github.com/zengarden/TinyExp/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/zengarden/TinyExp/branch/main/graph/badge.svg)](https://codecov.io/gh/zengarden/TinyExp)
[![Commit activity](https://img.shields.io/github/commit-activity/m/zengarden/TinyExp)](https://img.shields.io/github/commit-activity/m/zengarden/TinyExp)
[![License](https://img.shields.io/github/license/zengarden/TinyExp)](https://img.shields.io/github/license/zengarden/TinyExp)

A simple Python project for deep learning experiment management. It uses Ray for core distributed environment and backend setup, and provides basic, no-frills tracking for models, optimizers, and LR schedulers.

# Usage


```
pip install tinyexp
```

Run mnist example, By default, all available GPUs will be used.

```
import tinyexp
from tinyexp.examples.mnist_exp import Config
tinyexp.ConfigStore.instance().store(name="cfg", node=Config)
tinyexp.simple_ray_launch_exp()
```

# More Examples

1. ImageNet ResNet-50 Example with Extremely Fast Data Loading:

```bash
export IMAGENET_HOME=yours_imagenet_dir

import tinyexp
from tinyexp.examples.resnet_exp import Config
tinyexp.ConfigStore.instance().store(name="cfg", node=Config)
tinyexp.simple_ray_launch_exp()
```

# Develop


1. prepare env
```bash
# 1. clone repo
git clone https://github.com/zengarden/tinyexp.git
# 2. Set Up Your Development Environment, This will also generate your `uv.lock` file
make install
source .venv/bin/activate
```

2. After development, checking whether the code is standardized

```bash
# Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:
uv run pre-commit run -a
```
