from setuptools import setup, find_packages, Extension

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required_list = f.read().splitlines()

cmodule = Extension('host/src/host_clib',
                    sources=['host/src/host_clib.c'],
                    extra_compile_args=["-O3"])

setup(
    name="host-picker",
    version="2.4.3",
    author="Matteo Bagagli",
    author_email="matteo.bagagli@erdw.ethz.com",
    url="https://github.com/mbagagli/host-picker",
    description="High Order STatisics picking algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.5',
    install_requires=required_list,
    packages=find_packages(),
    package_data={"host": ['src/*.c']},
    include_package_data=True,
    ext_modules=[cmodule],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
    ],
)
