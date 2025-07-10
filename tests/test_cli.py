from gnfs import gnfs_factor
import cli


def test_cli_main_output(capsys):
    cli.main(["10"])
    captured = capsys.readouterr()
    assert "Factors of 10: [2, 5]" in captured.out
