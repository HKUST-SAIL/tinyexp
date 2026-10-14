"""Training-engine pieces that experiments call from inside ``Exp.run()``.

Currently a single subpackage: ``tinyexp.tiny_engine.accelerator`` (CPU / DDP / FSDP / TP / Hugging Face
backends behind one ``AcceleratorProtocol``). Import from there, e.g.
``from tinyexp.tiny_engine.accelerator import AcceleratorProtocol, DDPAccelerator``.
"""
