# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/zurutech/anomaly-toolbox/issues.

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

Anomaly Toolbox could always use more documentation, whether as
part of the official Anomaly Toolbox docs, in docstrings,
or even on the web in blog posts, articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/zurutech/anomaly-toolbox/issues.

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `anomaly_toolbox` for local development.

1. Clone `anomaly-toolbox` from the internal GitHub:

    ```
    $ git clone ssh://git@github.com:zurutech/anomaly-toolbox.git
    ```

2. Install your local copy into a virtualenv. Assuming you have [virtualenvwrapper]
   installed, this is how you set up your fork for local development:

    ```
    $ mkvirtualenv anomaly-toolbox
    $ cd anomaly-toolbox/
    $ pip install -e .
    ```

3. To get the dev toolchain just pip install the provided requirements into your virtualenv.
   **NOTE:** following [zurutech/styleguide] requirements are defined in `requirements.in`
   folder as text files compliant with the `.in` format specified by [pip-tools].
   Pinned versions are generated using [reqompyler].

    ```
    $ pip install -r requirements.txt
    ```

4. Create a branch for local development:

    ```
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass [flake8], [pylint],
   [black] and the tests, including testing other Python versions with [tox]:

    - Automatically run the pipeline with `tox`.

        ```console
        $ tox
        ```

        **Note:** `tox` can be run parallely with `tox -p auto -o`

    - Manually run them without `tox` **NOT RECOMMENDED**:

        ```console
        $ pytest tests --doctest-modules --cov=anomaly_toolbox --cov-report term-missing
        $ black anomaly_toolbox tests
        $ isort anomaly_toolbox tests
        $ flake8 anomaly_toolbox tests
        $ pylint anomaly_toolbox tests
        ```

6. Commit your changes and push your branch to GitHub:

    ```console
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
    ```

7. Submit a pull request through the GitHub website.


## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.
3. The pull request should work for the specified Python Versions.
4. If you have made change to the CI Pipeline test them locally

## Tips

To run a subset of tests:

$ pytest -x -s -vvv --doctest-modules WHAT_MODULE/TEST_SUBSET_TO_TEST --cov=ashpy


## Deploying

1. Manually bump the version in `src/anomaly_toolbox/__init__.py`.
2. Tag your version by prepending it with `v`. I.E., `git tag v1.1.0`.
3. Push your tags `$ git push --tags`.

Travis will then deploy to PyPI if tests pass.

<!-- Links -->
[black]: https://github.com/psf/black
[flake8-bugbear]: https://github.com/PyCQA/flake8-bugbear
[flake8]: https://github.com/PyCQA/flake8
[pip-tools]: https://github.com/jazzband/pip-tools
[pylint]: https://github.com/PyCQA/pylint
[pytest-cov]: https://github.com/pytest-dev/pytest-cov
[pytest]: https://github.com/pytest-dev/pytest
[reqompyler]: https://github.com/zurutech/reqompyler
[tox]: https://github.com/tox-dev/tox
[virtualenvwrapper]: https://virtualenvwrapper.readthedocs.io/en/master/
[zurutech/styleguide]: https://github.com/zurutech/styleguide/python.md
