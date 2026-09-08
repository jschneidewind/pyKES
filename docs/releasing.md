# Releasing pyKES

This document describes the release process for publishing pyKES to TestPyPI and PyPI.

## 1. Pre-release checklist

1. Update version in [pyproject.toml](../pyproject.toml).
2. Update [CHANGELOG.md](../CHANGELOG.md).
3. Ensure tests pass locally:

```bash
pytest
```

4. Build and validate distributions locally:

```bash
rm -rf build dist *.egg-info src/*.egg-info
uv build
uv publish --dry-run --no-attestations dist/*
```

5. Verify install from wheel in a clean environment:

```bash
python -m venv /tmp/pykes-release-test
source /tmp/pykes-release-test/bin/activate
pip install dist/*.whl
python -c "import pyKES; print('pyKES import OK')"
```

## 2. Tag strategy

- TestPyPI prerelease tags: `vX.Y.ZrcN` (example: `v0.1.1rc1`)
- PyPI production tags: `vX.Y.Z` (example: `v0.1.1`)

The GitHub workflow enforces `tag == v<project.version>` for production and also
accepts prerelease tags in the form `v<project.version>rcN`.

## 3. Configure Trusted Publishing

Configure a Trusted Publisher in both TestPyPI and PyPI:

- Owner: `jschneidewind`
- Repository: `pyKES`
- Workflow: `release-pypi.yml`
- Environment (TestPyPI): `testpypi`
- Environment (PyPI): `pypi`

No API token is required when Trusted Publishing is configured correctly.

## 4. Publish to TestPyPI

1. Create and push a prerelease tag:

```bash
git tag v0.1.1rc1
git push origin v0.1.1rc1
```

2. Wait for workflow completion in GitHub Actions.
3. Verify installation from TestPyPI:

```bash
python -m venv /tmp/pykes-testpypi
source /tmp/pykes-testpypi/bin/activate
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pyKES==0.1.1rc1
```

## 5. Publish to PyPI

1. Create and push the release tag:

```bash
git tag v0.1.1
git push origin v0.1.1
```

2. Confirm the `publish-pypi` workflow job succeeds.
3. Verify installation from PyPI:

```bash
python -m venv /tmp/pykes-pypi
source /tmp/pykes-pypi/bin/activate
pip install pyKES==0.1.1
```

## 6. What a tag now builds

A `v*` tag triggers **two** workflows, and they are independent:

* `release-pypi.yml` publishes the wheel, which is the library release.
* `container-image.yml` runs the tests, then builds and pushes the image the
  droplet actually runs, and prints the tag and digest to paste into the
  deployment's `.env`.

The server runs the **image**, not the wheel. That distinction is not
cosmetic: version 0.2.4 on PyPI was released before any of the database work
existed, so `pip install pyKES` at that version gets a tree with no
`database_app` in it. See [deployment.md](deployment.md).

The image workflow gates on the test suite; the PyPI workflow does not, and
adding `needs: [tests]` to it would be a reasonable next change.

---

## 7. Post-release

1. Create a GitHub Release for the tag.
2. Copy key notes from [CHANGELOG.md](../CHANGELOG.md).
3. Open a follow-up issue for the next version milestone.
