# Dependency Regression Investigation

## Scope and baseline

- **Branch tested:** `feature/merge_deps_removal`
- **Baseline:** `9c182a420109b311cca738ca74f0b36933f419ea`
- **Upstream ref:** `upstream/main` at `7e766c1adea9915794914bde8d640d4b1667ee0c`
- **Comparison base:** `9b24385f75d7dcb88cfc9b38751d5fb03b4142c2`
- **Test runtime:** Docker `python:3.13-slim` (Python 3.13.14), with `libgomp1` installed for the LightGBM wheel.
- **Installation:** `uv pip install --system --no-cache ".[test]"`, followed by `python -m pip check`.

The repository Dockerfiles were not used for checkout validation because they install the published PyCaret package rather than the tested source tree. Each upstream change was instead applied without committing in a disposable worktree based on the baseline and tested independently, not as the cumulative upstream series. Where a commit conflicted with this branch's Python-3.13 dependency markers, resolution retained those existing markers and applied only the upstream removal.

## Candidate inventory

| Upstream commit | Upstream change only | Result | Recommendation |
| --- | --- | --- | --- |
| `635416f0` | Reworks CI around `uv`, adds `pytest-timeout`, moves the `daal4py` import into clustering tests. | Superseded / do not cherry-pick wholesale. This branch already uses `uv` and has a newer `pytest.importorskip("daal4py")` test guard. | Keep the local CI/test implementation. |
| `345f9581` | Removes declared `numba`, removes it from `_show_versions`, and deletes the Numba-disable fixture from `tests/test_models.py`. | **Regression.** | Do not merge unchanged. |
| `8ad0d2a0` | Removes declared `markupsafe`, `importlib_metadata`, and `nbformat`. | Passed documented targeted checks. | Candidate for merge; cumulative/full-matrix validation remains unverified. |
| `b1727ca1` | Removes `wurlitzer`, removes its version report entry, and drops C/child-process output capture. | Passed documented targeted checks. | Candidate for merge; cumulative/full-matrix validation remains unverified. |
| `7e766c1a` | Removes declared `deprecation`. | Passed documented targeted checks. | Candidate for merge; cumulative/full-matrix validation remains unverified. |

`805629fb` only removes the stale-issue workflow and is not a dependency-removal candidate. Open upstream PR #26 was excluded because it is not merged upstream.

## Baseline result

The baseline passed:

```text
python --version                         # Python 3.13.14
uv pip install --system ".[test]"         # success
python -m pip check                      # No broken requirements found
python -m pytest -q tests/test_models.py # 5 passed
```

Running the same container without `libgomp1` fails before tests when importing LightGBM (`OSError: libgomp.so.1`). This is a base-image prerequisite, not a candidate regression.

## Per-commit test results

### `345f9581` — remove `numba`

The package metadata no longer declared Numba and `pip check` passed. Numba was still installed transitively by PyOD, which means this direct-requirement removal was not shown to eliminate Numba from the resolved environment.

```text
python -m pytest -q tests/test_models.py::test_model_equality_anomaly
# FAILED: numba.core.errors.TypingError in pyod.models.abod._wcos
```

The baseline's same test passes because its fixture disables Numba JIT and routes dispatchers to their Python functions. Deleting that fixture exposes a Numba/NumPy JIT failure while fitting PyOD ABOD. Therefore `345f9581` introduces a **test regression through fixture removal**, not through proof that its direct dependency removal uninstalls Numba. Do not merge it unchanged; retain an explicit test-time Numba dependency for the fixture or replace the fixture with an optional-import workaround.

### `8ad0d2a0` — remove notebook/Colab packages

```text
python -m pip check                      # passed
python -c 'import pycaret'               # passed
python -m pytest -q tests/test_models.py # 5 passed
```

The installed distribution no longer declares `markupsafe`, `importlib_metadata`, or `nbformat`. Some are still installed transitively by remaining dependencies, which is compatible with removing them from PyCaret's direct requirements.

`tests/test_create_docker.py` failed under the deliberately limited `.[test]` installation because it calls `create_api`, which requires `fastapi` from the `mlops` extra. It was out of scope for this candidate's documented targeted test matrix and was not baseline-reproduced in the same limited environment, so it is not classified as a candidate regression.

### `b1727ca1` — remove `wurlitzer`

```text
python -m pip check                                                  # passed
redirect_output().__enter__(); redirect_output().__exit__(...)      # passed
python -m pytest -q tests/test_models.py::test_model_equality_classification
# 1 passed
```

The distribution no longer declares `wurlitzer`, and the Python-level output redirect context remains usable. The upstream change intentionally stops capturing stdout/stderr emitted by C libraries and child processes; that behavioral reduction is the remaining risk to assess for logging-sensitive workflows.

### `7e766c1a` — remove `deprecation`

```text
python -m pip check            # passed
python -c 'import pycaret'     # passed
```

The installed PyCaret metadata no longer declares `deprecation`, and importing PyCaret succeeds on Python 3.13.

## Final recommendation

1. **Candidates for merge with the branch's existing Python-3.13 markers preserved:** `8ad0d2a0`, `b1727ca1`, and `7e766c1a`. They passed independent targeted checks only; run cumulative/full-matrix validation before merging.
2. **Do not merge `345f9581` unchanged.** Its deleted fixture is currently required to avoid the PyOD/Numba ABOD failure.
3. **Do not wholesale cherry-pick `635416f0`.** Its CI and `daal4py` changes are already superseded by this branch's corresponding configuration and test guard.
4. Before merging the safe removals, run the repository's full CI matrix with the project's full extras; this investigation intentionally used dependency-focused Python-3.13 Docker tests.
