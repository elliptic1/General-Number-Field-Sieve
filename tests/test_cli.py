from gnfs import gnfs_factor
import cli


def test_cli_main_output(capsys):
    cli.main(["10", "--degree", "1", "--bound", "30", "--interval", "50"])
    captured = capsys.readouterr()
    assert "Factors of 10: [2, 5]" in captured.out
