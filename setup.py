# Copyright 2021 Zuru Tech HK Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        scripts=["bin/anomaly-box.py"],
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        url="https://github.com/zurutech/anomaly-toolbox",
        version="0.1.0",
        zip_safe=False,
    )


if __name__ == "__main__":
    sys.exit(main())
