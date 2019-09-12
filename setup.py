#!/usr/bin/env python

from setuptools import setup
import os

def get_version():
    with open("dynamix/__init__.py", "r") as fid:
        lines = fid.readlines()
    version = None
    for line in lines:
        if "version" in line:
            version = line.rstrip().split("=")[-1].lstrip()
    if version is None:
        raise RuntimeError("Could not find version from __init__.py")
    version = version.strip("'").strip('"')
    return version


def setup_package():

    version = get_version()

    packages_folders = [
        "correlator",
        "resources",
    ]
    packages = ["dynamix", "dynamix.test"]
    package_dir = {"dynamix": "dynamix",
                   "dynamix.test": "dynamix/test"}
    for f in packages_folders:
        modulename = str("dynamix.%s" % f)
        packages.append(modulename)
        package_dir[modulename] = os.path.join("dynamix", f)
        module_test_dirname = os.path.join(package_dir[modulename], "test")
        if os.path.isdir(module_test_dirname):
            modulename_test = str("%s.test" % modulename)
            packages.append(modulename_test)
            package_dir[modulename_test] = module_test_dirname
    setup(
        name='dynamix',
        author='Pierre Paleo, Jerome Kieffer',
        maintainer='Pierre Paleo, Jerome Kieffer',
        version=version,
        author_email = "pierre.paleo@esrf.fr",
        maintainer_email = "pierre.paleo@esrf.fr",
        url='https://github.com/silx-kit/dynamix',

        packages=packages,
        package_dir = package_dir,
        package_data = {
            'dynamix.resources': [
                'opencl/*.cl',
            ]
        },
        install_requires = [
          'numpy',
          'pyopencl',
        ],
        long_description = """
        dynamix - software for X-ray photon correlation spectroscopy
        """,

        zip_safe=True
    )


if __name__ == "__main__":
    setup_package()
