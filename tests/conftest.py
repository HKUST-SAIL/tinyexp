import pytest
import ray


@pytest.fixture(scope="session")
def ray_session():
    """
    Initializes Ray for the entire test session and shuts it down afterwards.
    This ensures Ray is started only once, making tests faster and more stable.
    """
    if not ray.is_initialized():
        runtime_env = {
            "working_dir": ".",
            "excludes": ["*.md", "data/", "tests/", ".git/", ".venv/", "output/", "outputs/", "site/"],
        }
        try:
            ray.init(runtime_env=runtime_env)
        except Exception as exc:
            pytest.skip(f"Ray is not available in this environment: {exc}")

    yield

    ray.shutdown()
