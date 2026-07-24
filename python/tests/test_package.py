"""Python-side test scaffolding.

Real tests arrive in Step 2 (Gymnasium env over the nanobind module). For now
this verifies the package imports and the pytest harness runs — so `poe pytest`
is wired end-to-end from day one.
"""

import md


def test_package_imports_and_reports_version():
    assert md.__version__ == "0.1.0"
