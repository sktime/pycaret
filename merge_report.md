# Upstream Merge Report

## Summary

`feature/merge_deps_removal` merged `upstream/main` at merge commit `58c9cff1`. This establishes shared ancestry through upstream commit `7e766c1a`, preventing the same six upstream commits from being re-integrated as a future divergence.

The merge preserved this branch's existing Python 3.13 CI matrices and `daal4py` guard. Two deliberate follow-up commits restore behavior that upstream removed but this fork requires:

- `e638bea7` — `bug: retain numba test safeguard`
- `5f45a6bf` — `bug: retain wurlitzer logging`

## Upstream commit disposition

| Upstream commit | Subject | Decision | Local outcome and rationale |
| --- | --- | --- | --- |
| `805629fb` | `[MNT] remove stale.yml workflow which automatically closes issues or PR (#17)` | Kept | The stale issue/PR workflow was removed as upstream intended. |
| `635416f0` | `[MNT] fix CI - use uv, remove legacy builds (#15)` | Adapted | Its upstream ancestry is retained. The branch kept its already-working Python 3.13 CI matrix and stronger `pytest.importorskip("daal4py")`/macOS guard rather than replacing them with the older workflow and import changes. The upstream test extra addition (`pytest-timeout`) is retained through the merge. |
| `345f9581` | `[MNT] remove unnecessary dependencies - numba (#23)` | Rewritten | The upstream removal was accepted by the merge, then `e638bea7` restored Numba's Python-version dependency declarations, version reporting, and the `disable_numba` test fixture. Docker Python 3.13 testing showed that deleting this fixture causes PyOD ABOD to raise `numba.core.errors.TypingError`; keeping it prevents that regression. |
| `8ad0d2a0` | `[MNT] remove unnecessary deps - notebook/colab related packages (#22)` | Kept | Removed direct `markupsafe`, `importlib_metadata`, and `nbformat` requirements while retaining the branch's Python 3.13 dependency markers. Targeted Docker tests passed. |
| `b1727ca1` | `[MNT] remove unnecessary deps - wurlitzer (#21)` | Rewritten | The upstream removal was accepted by the merge, then `5f45a6bf` restored `wurlitzer`, its version report entry, and `redirect_output`'s C/child-process stdout/stderr capture. This fork intentionally keeps that logging capability. |
| `7e766c1a` | `[MNT] Remove unused packages - deprecations (#25)` | Kept | Removed the direct `deprecation` requirement. Targeted Docker install/import checks passed. |

## Conflict resolutions

| File | Resolution |
| --- | --- |
| `.github/workflows/test.yml` | Kept the local workflow because it already uses `uv` and contains the branch's Python 3.13-focused CI configuration. |
| `tests/test_clustering_engines.py` | Kept the local `pytest.importorskip("daal4py")`, macOS/Python-3.13 guard, and module-based assertions. |
| `pyproject.toml` | Retained local Python-version markers. Applied safe upstream removals; restored Numba and Wurlitzer through the separate override commits. |

## Validation

The integrated branch was tested in Docker using `python:3.13-slim` with `libgomp1` installed for LightGBM:

```text
Python 3.13.14
uv pip install --system --no-cache ".[test]"  # passed
python -m pip check                           # passed
redirect_output() context smoke test          # passed
python -m pytest -q tests/test_models.py      # 5 passed
```

The full CI matrix was not run locally. The Docker validation specifically confirms the two protected regressions (Numba fixture and Wurlitzer logging) alongside the dependency merge.
