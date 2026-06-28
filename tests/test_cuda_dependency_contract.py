from pathlib import Path


def test_cuda_transformers_floor_supports_dtype_loader_api():
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    contents = pyproject.read_text(encoding="utf-8")
    cuda_dependencies = contents.split("cuda = [", 1)[1].split("]", 1)[0]

    assert '"transformers>=4.56.0,<5"' in cuda_dependencies
