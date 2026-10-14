# TinyExp examples

Each example is one runnable file. Its module docstring starts with a run header — `Modes`, `Run`, `Env`,
`Needs`, `Out` — which is the single source of commands for that example (kept valid by
`tests/examples/test_examples_help_smoke.py`). Read the header, not the code, to run an example:

```bash
python -c "import tinyexp.examples.mnist_exp as m; print(m.__doc__)"
```

| You want to... | Start from | Data | Hardware |
| --- | --- | --- | --- |
| Write a new training experiment (template) | `mnist_exp.py` | auto-download | CPU, or GPUs with DDP |
| Run a non-deep-learning distributed job / fastest smoke test | `pi_exp.py` | none | CPU |
| Train on ImageNet with DDP, FSDP or HF Accelerate; Redis byte cache | `resnet_exp.py` | `IMAGENET_HOME` | GPUs |
| Use tensor parallelism; reproduce DeiT-S; benchmark TP vs DDP | `vit_tp_exp.py` (+ `docs/vit_tp.md`) | `IMAGENET_HOME` | 2+ GPUs (CPU smoke available) |

All examples share the same shape: nested `XXXCfg` dataclasses (each with `build_*` helpers) followed by `run()`,
which dispatches on `mode`. Check any override with `mode=help` before running; outputs go to
`<output_root>/<exp_name>/`.
