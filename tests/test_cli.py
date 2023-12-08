import subprocess

def cli_check_success(args):
    """
    Shared helper function to call shell commands.
    """
    subprocess.check_call(args)

def test_cross_lines_z():
    cli_check_success(['cross_lines_z', '-h'])

def test_profile_extraction():
    cli_check_success(['profile_extraction', '-h'])

def test_burn_line_z():
    cli_check_success(['burn_line_z', '-h'])
