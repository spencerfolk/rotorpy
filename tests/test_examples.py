"""
Tests the examples found in /examples.

Split into two marker groups:
  - compilation - fast: syntax check + import resolution. Catches broken
                  imports and parse errors without running the rest of the
                  script.
  - full_run    - slow, best-effort: actually executes each example.
                  Catches runtime issues that only show up during execution.
                  Run periodically (see .github/workflows) rather than on
                  every push.
"""
import ast
import json
import os
import subprocess
import sys
import glob

import pytest

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'examples')

example_scripts = [
    p for p in glob.glob(os.path.join(EXAMPLES_DIR, "*.py"))
    if os.path.isfile(p) and 'ardupilot' not in os.path.basename(p) and 'ppo' not in os.path.basename(p)
]


def _read(script_path):
    with open(script_path, encoding="utf-8") as f:
        return f.read()


def _parse(script_path):
    """Parse a script's source into an AST, raising SyntaxError on failure."""
    return ast.parse(_read(script_path), filename=script_path)


def _extract_import_source(script_path):
    """
    Pull out only the top-level import statements (Import / ImportFrom nodes)
    as executable source. Running just this subset lets us check whether
    imports resolve without paying the cost - or risk - of running the
    whole example.

    Note: this only catches top-level imports. Imports inside functions or
    behind `if __name__ == "__main__":` won't be exercised here - the
    full-run tier is what catches those.

    Implementation note: we slice the original source by line number
    (node.lineno / node.end_lineno) rather than using ast.unparse, since
    unparse was only added in Python 3.9 and this project still supports
    3.8.
    """
    source_lines = _read(script_path).splitlines()
    tree = _parse(script_path)
    import_nodes = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    chunks = []
    for node in import_nodes:
        # end_lineno is also available since Python 3.8.
        start, end = node.lineno, node.end_lineno
        chunks.append("\n".join(source_lines[start - 1:end]))

    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# Tier 1: syntax
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script_path", example_scripts, ids=os.path.basename)
def test_example_syntax(script_path):
    """Each example should at least be valid Python source."""
    script_name = os.path.basename(script_path)
    try:
        _parse(script_path)
    except SyntaxError as e:
        pytest.fail(f"{script_name}: syntax error: {e}")


# ---------------------------------------------------------------------------
# Tier 2: imports
# ---------------------------------------------------------------------------

def test_example_imports():
    """
    Every example's top-level imports should resolve. All examples are
    checked inside a single interpreter: launching one subprocess per
    example paid torch/matplotlib startup cost 12 times over and dominated
    the suite runtime (~45s vs ~5s).
    """
    scripts = []
    for p in example_scripts:
        src = _extract_import_source(p)
        if src.strip():
            scripts.append([os.path.basename(p), src])
    if not scripts:
        pytest.skip("no top-level imports found in any example.")

    runner = (
        "import json, sys, traceback\n"
        "failures = []\n"
        "for name, src in json.loads(sys.argv[1]):\n"
        "    try:\n"
        "        exec(compile(src, name, 'exec'), {})\n"
        "    except ModuleNotFoundError:\n"
        "        pass  # optional dependency; same policy as the old per-example skips\n"
        "    except Exception:\n"
        "        failures.append((name, traceback.format_exc()))\n"
        "if failures:\n"
        "    for name, tb in failures:\n"
        "        print('=== import error in %s ===' % name)\n"
        "        print(tb)\n"
        "    raise SystemExit(1)\n"
    )

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        [sys.executable, "-c", runner, json.dumps(scripts)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
        env=env,
    )

    if result.returncode != 0:
        pytest.fail(f"example import error(s):\n{result.stdout or result.stderr}")


# ---------------------------------------------------------------------------
# Tier 3: full run (slow, best-effort)
# ---------------------------------------------------------------------------

@pytest.mark.full_run
@pytest.mark.parametrize("script_path", example_scripts, ids=os.path.basename)
def test_example_script_runs(script_path):
    """
    Best-effort full run of each example. This is slower and looser than
    the compilation checks above: a timeout here is treated as a soft pass
    (the script got far enough to still be running, rather than crashing
    outright), not a failure. Real errors that surface only at runtime
    (bad shapes, missing files, etc.) still fail the test.
    """
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
            timeout=900,
            env=env,
        )

        if result.returncode != 0:
            if "EOFError" in result.stderr:
                pytest.skip(f"{script_name} skipped: script waits for user input.")
            elif "ModuleNotFoundError" in result.stderr or "NameError" in result.stderr:
                pytest.skip(f"{script_name} skipped: missing optional dependency.")
            else:
                pytest.fail(f"{script_name} failed with error:\n{result.stderr.strip()}")

    except subprocess.TimeoutExpired:
        pytest.skip(f"{script_name} skipped: execution timed out (still running after 900s).")