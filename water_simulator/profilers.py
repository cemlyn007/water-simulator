import contextlib
import ctypes

try:
    _cuda = ctypes.CDLL("libcudart.so")
except OSError:
    _cuda = None


@contextlib.contextmanager
def nvidia_profile():
    if _cuda is None:
        raise ImportError("libcudart.so not found")
    # else...
    _cuda.cudaProfilerStart()
    yield
    _cuda.cudaProfilerStop()
