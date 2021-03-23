#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `anomaly_toolbox` package."""

import pytest

from click.testing import CliRunner

from anomaly_toolbox import anomaly_toolbox
from anomaly_toolbox import cli


@pytest.fixture
def meaning_of_life():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    return 42


def test_content(meaning_of_life):
    """Sample pytest test function with the pytest fixture as an argument."""
    assert 42 == meaning_of_life


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "anomaly_toolbox.cli.main" in result.output
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output
