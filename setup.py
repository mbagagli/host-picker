from setuptools import setup, Extension

cmodule = Extension(
    name="host.src.host_clib",        # -> host/src/host_clib.*.so
    sources=["host/src/host_clib.c"],
    extra_compile_args=["-O3"],
)

setup(
    ext_modules=[cmodule],
)
