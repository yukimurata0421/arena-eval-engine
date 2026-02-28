from arena import cli


def test_cli_validate_runs():
    code = cli.main(["validate"])
    assert code in (0, 1)
