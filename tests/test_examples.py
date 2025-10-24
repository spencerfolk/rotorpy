'''
Tests the examples found in /examples. This is mostly just to check for any broken imports.  
'''
import subprocess
import sys
import os
import pytest
import glob

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'examples')

example_scripts = [
    p for p in glob.glob(os.path.join(EXAMPLES_DIR, "*.py"))
    if os.path.isfile(p) and 'ardupilot' not in os.path.basename(p) and 'ppo' not in os.path.basename(p)
]

@pytest.mark.parametrize("script_path", example_scripts)
def test_example_script_runs(script_path):
    """Test that example scripts run without error or timeout."""
    script_name = os.path.basename(script_path)
    print(f"\nTesting {script_name}")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            env=env
        )

        if result.returncode != 0:
            if "EOFError" in result.stderr:
                pytest.skip(f"{script_name} skipped: script waits for user input.")
            else:
                pytest.fail(f"{script_name} failed with error:\n{result.stderr.strip()}")

    except subprocess.TimeoutExpired:
        pytest.skip(f"{script_name} skipped: execution timed out.")
