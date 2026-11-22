"""Stand-in for pycuda.driver: importable, but there is no device."""


class Error(Exception):
    pass


def init():
    raise Error("stand-in PyCUDA: no CUDA device")
