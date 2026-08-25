from onyx_cuda import _rust


def test_native_extension_imports():
    assert _rust.__name__ == "onyx_cuda._rust"
