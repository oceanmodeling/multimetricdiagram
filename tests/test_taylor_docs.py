import doctest

import multimetricdiagram as mmd


def test_docstring_example_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (test,) = doctest.DocTestFinder(recurse=False).find(mmd.alt.taylor, globs={})
    result = doctest.DocTestRunner().run(test)
    assert result.attempted == 11 and result.failed == 0
    assert (tmp_path / "taylor.png").stat().st_size > 0
