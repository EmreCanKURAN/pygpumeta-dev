# Releasing

Releases are published by `.github/workflows/release.yml` with PyPI trusted
publishing: PyPI accepts uploads from this repository's workflow directly,
and no API token is stored anywhere.

## One-time setup

1. On [TestPyPI](https://test.pypi.org/manage/account/publishing/) and on
   [PyPI](https://pypi.org/manage/account/publishing/), add a pending
   trusted publisher for the project name `pygpumeta` with

   - owner `EmreCanKURAN`, repository `pygpumeta-dev`,
   - workflow `release.yml`,
   - environment `testpypi` on TestPyPI and `pypi` on PyPI.

2. In the repository settings on GitHub, create the environments `testpypi`
   and `pypi`. Give `pypi` a required reviewer, so that nothing reaches PyPI
   without an explicit approval.

## Making a release

1. Run the GPU tier on a machine with a CUDA device
   (`python -m pytest -m gpu`) and `python -m pygpumeta.doctor`. CI has no
   GPU and cannot do this.
2. Set the version in `src/pygpumeta/_version.py`, move the "Unreleased"
   entries of `CHANGELOG.md` under that version and date, and merge.
3. Rehearse: start the workflow by hand (Actions > Release > Run workflow).
   It builds, tests the wheel in a clean environment and publishes to
   TestPyPI only. Check the result with

   ```
   pip install --no-deps --index-url https://test.pypi.org/simple/ pygpumeta
   ```

   TestPyPI does not accept the same version twice; use a development
   version such as `0.1.0.dev1` for rehearsals.
4. Create a GitHub release with the tag `v<version>`. The workflow refuses
   to continue if the tag and the version differ, publishes to TestPyPI and
   waits for the approval of the `pypi` environment before publishing to
   PyPI.
