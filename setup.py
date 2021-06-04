"""The setup script."""

import sys

from setuptools import find_packages, setup


def main():
    """Setup script."""

    with open("README.md") as readme_file:
        readme = readme_file.read()

    requirements = open("requirements.in").read().split()

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
        install_requires=requirements,
        python_requires=">=3.8",
        license="MIT license",
        long_description=readme,
        long_description_content_type="text/markdown",
        include_package_data=True,
        keywords="anomaly_toolbox",
        name="anomaly_toolbox",
        scripts=["bin/anomaly-box.py"],  # TODO: rename and remove .py
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        url="https://github.com/zurutech/anomaly-toolbox",
        version="0.1.0",
        zip_safe=False,
    )


if __name__ == "__main__":
    sys.exit(main())
