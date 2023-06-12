#!/usr/bin/env python
import distutils.command.clean
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).parent.resolve()


def _run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, cwd=ROOT_DIR, stderr=subprocess.DEVNULL).decode("ascii").strip()
    except Exception:
        return None


def _get_version(sha):
    with open(ROOT_DIR / "version.txt", "r") as f:
        version = f.read().strip()
    if os.getenv("BUILD_VERSION"):
        version = os.getenv("BUILD_VERSION")
    elif sha is not None:
        version += "+" + sha[:7]
    return version


def _make_version_file(version, sha):
    sha = "Unknown" if sha is None else sha
    version_path = ROOT_DIR / "torchtime" / "version.py"
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = '{sha}'\n")


def _get_pytorch_version():
    if "PYTORCH_VERSION" in os.environ:
        return f"torch=={os.environ['PYTORCH_VERSION']}"
    return "torch"


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove torchtime extension
        for path in (ROOT_DIR / "torchtime").glob("**/*.so"):
            print(f"removing '{path}'")
            path.unlink()
        # Remove build directory
        build_dirs = [
            ROOT_DIR / "build",
        ]
        for path in build_dirs:
            if path.exists():
                print(f"removing '{path}' (and everything under it)")
                shutil.rmtree(str(path), ignore_errors=True)


def _get_packages(branch_name, tag):
    exclude = [
        "build*",
        "test*",
        "third_party*",
        "tools*",
    ]
    exclude_prototype = False
    if branch_name is not None and branch_name.startswith("release/"):
        exclude_prototype = True
    if tag is not None and re.match(r"v[\d.]+(-rc\d+)?", tag):
        exclude_prototype = True
    if exclude_prototype:
        print("Excluding torchtime.prototype from the package.")
        exclude.append("torchtime.prototype*")
    return find_packages(exclude=exclude)


def _init_submodule():
    print(" --- Initializing submodules")
    try:
        subprocess.check_call(["git", "submodule", "init"])
        subprocess.check_call(["git", "submodule", "update"])
    except Exception:
        print(" --- Submodule initalization failed")
        print("Please run:\n\tgit submodule update --init --recursive")
        sys.exit(1)
    print(" --- Initialized submodule")


def _parse_url(path):
    with open(path, "r") as file_:
        for line in file_:
            match = re.match(r"^\s*URL\s+(https:\/\/.+)$", line)
            if match:
                url = match.group(1)
                yield url


def _parse_sources():
    third_party_dir = ROOT_DIR / "third_party"
    libs = ["sox"]
    archive_dir = third_party_dir / "archives"
    archive_dir.mkdir(exist_ok=True)
    for lib in libs:
        cmake_file = third_party_dir / lib / "CMakeLists.txt"
        for url in _parse_url(cmake_file):
            path = archive_dir / os.path.basename(url)
            yield path, url


def _main():
    sha = _run_cmd(["git", "rev-parse", "HEAD"])
    branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    tag = _run_cmd(["git", "describe", "--tags", "--exact-match", "@"])
    print("-- Git branch:", branch)
    print("-- Git SHA:", sha)
    print("-- Git tag:", tag)
    pytorch_package_dep = _get_pytorch_version()
    print("-- PyTorch dependency:", pytorch_package_dep)
    version = _get_version(sha)
    print("-- Building version", version)

    _make_version_file(version, sha)

    with open("README.md") as f:
        long_description = f.read()

    setup(
        name="torchtime",
        version=version,
        description="An time series package for PyTorch",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/VincentSch4rf/torchtime",
        author=(
            "Vincent Scharf"
        ),
        author_email="vincent.scharf@smail.inf.h-brs.de",
        maintainer="Vincent Scharf",
        maintainer_email="vincent.scharf@smail.inf.h-brs.de",
        classifiers=[
            "Environment :: Plugins",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Programming Language :: C++",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        packages=_get_packages(branch, tag),
        cmdclass={
            "clean": clean,
        },
        install_requires=[pytorch_package_dep],
        zip_safe=False,
    )


if __name__ == "__main__":
    _main()