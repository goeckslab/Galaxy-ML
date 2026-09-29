# Publishing Galaxy-ML to PyPI

Publishing a GitHub release triggers the `Publish to PyPI` workflow. Merging a
pull request does not publish to PyPI. The workflow uses PyPI Trusted Publishing
with GitHub Actions OIDC, without a PyPI API token.

## Publish a new version

1. Update `__version__` in `galaxy_ml/__init__.py` and merge the release changes
   into `main`.
2. Open the repository's **Releases** page and draft a new release. Create a
   matching `release_v<version>` tag pointing to the commit containing those
   changes.
3. Click **Publish release** to trigger the upload workflow.
4. Open **Actions → Publish to PyPI** and verify that the build and publish jobs
   succeed, then check the new version on [PyPI](https://pypi.org/project/Galaxy-ML/).

The workflow builds the tagged commit, verifies that the tag matches the package
version, checks the source archive's metadata, and verifies that its Cython
extensions compile. It uploads the source archive; the compiled wheel is used
only for validation. If PyPI already has files for that version, it skips the
upload.

The PyPI project description comes from `setup.py` in the tagged commit.
Changes to that description appear on PyPI when a version containing them is
uploaded.
