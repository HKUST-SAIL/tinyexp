[![Main](https://github.com/HKUST-SAIL/tinyexp/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/HKUST-SAIL/tinyexp/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/HKUST-SAIL/TinyExp/branch/main/graph/badge.svg)](https://codecov.io/gh/HKUST-SAIL/TinyExp)

# TinyExp

A simple Python project for deep learning experiment management.

TinyExp lets you launch experiments with one click: the file you edit becomes the entrypoint to your experiment.

# Usage

```
pip install tinyexp
```

1. Run mnist example(By default, It will trained on CPU)

```python
import tinyexp
from tinyexp.examples.mnist_exp import Exp, store_and_run_exp
store_and_run_exp(Exp)
```

2. or:

```
python tinyexp/examples/mnist_exp.py
```

3. Run mnist example with overridden config:

```
python tinyexp/examples/mnist_exp.py dataloader_cfg.train_batch_size_per_device=16
```

see all available configs:

```
python tinyexp/examples/mnist_exp.py mode=help
```

see all available configs with overridden configs:

```
python tinyexp/examples/mnist_exp.py mode=help dataloader_cfg.train_batch_size_per_device=16
```


# More Examples

1. ImageNet ResNet-50 Example with Extremely Fast Data Loading (By default, all available GPUs will be used.)

```python
# export IMAGENET_HOME=yours_imagenet_dir

import tinyexp
from tinyexp.examples.resnet_exp import ResNetExp, store_and_run_exp
store_and_run_exp(ResNetExp)
```

# Develop

1. prepare env

```bash
# 1. clone repo
git clone https://github.com/HKUST-SAIL/tinyexp.git
# 2. Set Up Your Development Environment, This will also generate your `uv.lock` file
make install
source .venv/bin/activate
```

2. After development, checking whether the code is standardized

```bash
# Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:
uv run pre-commit run -a
```

# License

MIT License. See `LICENSE`.
