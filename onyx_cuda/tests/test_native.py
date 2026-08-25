from onyx_cuda import _rust


def test_native_extension_imports():
    assert _rust.version() == "0.1.0"
