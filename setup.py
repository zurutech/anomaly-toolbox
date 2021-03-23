#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    README = readme_file.read()

setup(
    author="Zuru Tech HK Limited, All rights reserved.",
    author_email="ml@zuru.tech",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
    ],
    description="Anomaly Toolbox Powered by GANs",
    entry_points={
        "console_scripts": [
            "anomaly_toolbox=anomaly_toolbox.cli:main",
        ],
    },
    install_requires=[
        "Click>=7.0",
    ],
    python_requires=">=3.7",
    license="MIT license",
    long_description=README,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="anomaly_toolbox",
    name="anomaly_toolbox",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    url="https://github.com/zurutech/anomaly-toolbox",
    version="0.1.0",
    zip_safe=False,
)
