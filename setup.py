#!/usr/bin/env python
import os
import os.path as osp
import subprocess
import time

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


def readme():
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    return content


MAJOR = 0
MINOR = 1
PATCH = 1
SUFFIX = ""
if PATCH != "":
    SHORT_VERSION = "{}.{}.{}{}".format(MAJOR, MINOR, PATCH, SUFFIX)
else:
    SHORT_VERSION = "{}.{}{}".format(MAJOR, MINOR, SUFFIX)

version_file = "openunreid/version.py"


def get_git_hash():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ["SYSTEMROOT", "PATH", "HOME"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        sha = out.strip().decode("ascii")
    except OSError:
        sha = "unknown"

    return sha


def get_hash():
    if os.path.exists(".git"):
        sha = get_git_hash()[:7]
    elif os.path.exists(version_file):
        try:
            from openunreid.version import __version__  # noqa

            sha = __version__.split("+")[-1]
        except ImportError:
            raise ImportError("Unable to get git version")
    else:
        sha = "unknown"

    return sha


def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}
__version__ = '{}'
short_version = '{}'
"""
    sha = get_hash()
    VERSION = SHORT_VERSION + "+" + sha

    with open(version_file, "w") as f:
        f.write(content.format(time.asctime(), VERSION, SHORT_VERSION))


def get_version():
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]


def numpy_include():
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    return numpy_include


ext_modules = [
    Extension(
        "openunreid.core.metrics.rank_cylib.rank_cy",
        ["openunreid/core/metrics/rank_cylib/rank_cy.pyx"],
        include_dirs=[numpy_include()],
    )
]


def get_requirements(filename="requirements.txt"):
    here = osp.dirname(osp.realpath(__file__))
    with open(osp.join(here, filename), "r") as f:
        requires = [line.replace("\n", "") for line in f.readlines()]
    return requires


if __name__ == "__main__":
    write_version_py()
    setup(
        name="openunreid",
        version=get_version(),
        description="Unsupervised (Domain Adaptive) Object Re-ID Toolbox and Benchmark",
        long_description=readme(),
        author="Yixiao Ge",
        author_email="geyixiao831@gmail.com",
        keywords="computer vision, unsupervised learning, domain adaptation, object re-ID",  # noqa
        url="https://github.com/open-mmlab/OpenUnReID",
        packages=find_packages(),
        license="Apache License 2.0",
        install_requires=get_requirements(),
        ext_modules=cythonize(ext_modules),
    )
