from tinyexp import TinyExp


def test_tiny_exp():
    class MyExperiment(TinyExp):
        pass

    _ = MyExperiment(cfg=None)
