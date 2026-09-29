# Publishing Galaxy-ML to PyPI

The `Publish to PyPI` workflow uses PyPI Trusted Publishing with GitHub Actions
OIDC. It does not require a PyPI API token or a GitHub Actions secret.

## One-time setup

1. Merge `.github/workflows/publish.yaml` into the repository's default branch.
2. Create a GitHub Actions environment named `pypi` under repository Settings
   → Environments. A required reviewer can be configured to approve uploads.
3. In PyPI, open Galaxy-ML → Manage → Publishing and add a GitHub publisher:

   | Field | Value |
   | --- | --- |
   | Owner | `goeckslab` |
   | Repository name | `Galaxy-ML` |
   | Workflow name | `publish.yaml` |
   | Environment name | `pypi` |

   The workflow name is the filename, not its display name or full path.
   If the Publishing settings are unavailable, ask the PyPI project owner
   to configure the publisher.

See the [PyPI Trusted Publishing setup guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

## New releases

Update `galaxy_ml/__init__.py` and create a matching `release_v<version>` tag.
Publish a GitHub release for that tag. This triggers the workflow, which checks
that the tag matches the package version, builds a source archive, checks its
metadata, and verifies that its Cython extensions compile from the archive.
Only the source archive is uploaded, matching the existing PyPI distribution
format. The compiled wheel is used for validation, not publication.

## Existing GitHub releases, including 0.11.0

Adding the workflow does not replay an earlier GitHub release event. To publish
an existing tag, go to Actions → Publish to PyPI → Run workflow, select `main`
for the workflow, and enter the tag, for example `release_v0.11.0`.
The package is built from that tag, not from the current default branch.

If PyPI already has files for that version, the workflow reports that it is
skipping the upload and does not build or publish it. It does not overwrite
files or complete a partially uploaded release. PyPI errors other than a
missing version stop the workflow. Concurrent runs for the same tag are
serialized; an upload race with an external publisher fails visibly.
