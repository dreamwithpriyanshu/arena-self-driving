#!/usr/bin/env python
"""
Smoke test for Build Step 4b — Streamlit UI.

Verifies:
1. Architecture boundary: Pages do not import src.envs, src.agents, or src.training.
2. Streamlit compilation: All pages load without Syntax/Import errors.
"""

from __future__ import annotations

import sys
import os
import ast
from pathlib import Path
from unittest.mock import MagicMock

# Force UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _header(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def test_ui_architecture() -> None:
    _header("Test 1: UI Architecture Bounds")
    
    pages_dir = Path("pages")
    app_py = Path("app.py")
    
    files_to_check = [app_py] + list(pages_dir.glob("*.py"))
    
    forbidden_modules = ["src.envs", "src.agents", "src.training"]
    
    for filepath in files_to_check:
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError as exc:
                raise ValueError(f"Syntax error in {filepath.name}: {exc}")
                
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for fm in forbidden_modules:
                        if alias.name.startswith(fm):
                            raise ValueError(f"{filepath.name} illegally imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for fm in forbidden_modules:
                        if node.module.startswith(fm):
                            raise ValueError(f"{filepath.name} illegally imports from {node.module}")
                            
    print("  [OK] Streamlit pages correctly respect the facade boundaries.")


from streamlit.testing.v1 import AppTest

def test_streamlit_compilation() -> None:
    _header("Test 2: Streamlit Headless Compilation")
    
    pages_dir = Path("pages")
    files_to_check = [Path("app.py")] + list(pages_dir.glob("*.py"))
    
    for filepath in files_to_check:
        try:
            at = AppTest.from_file(str(filepath.absolute()))
            at.run(timeout=10)
            if at.exception:
                raise RuntimeError(f"AppTest caught exception: {at.exception[0]}")
            print(f"  [OK] {filepath.name} ran headlessly successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to compile/execute {filepath.name}: {e}")


def main() -> int:
    print("=" * 60)
    print("  Build Step 4b Smoke Test -- Streamlit UI")
    print("=" * 60)

    tests = [
        test_ui_architecture,
        test_streamlit_compilation,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            failed += 1
            print(f"\n  [FAIL] {test_fn.__name__}")
            print(f"    {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()

    _header("SUMMARY")
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n  ALL SMOKE TESTS PASSED\n")
        return 0
    else:
        print("\n  SOME TESTS FAILED\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
