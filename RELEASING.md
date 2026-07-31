# Release process

Releases are built by GitHub Actions and published with PyPI Trusted
Publishing. Do not upload a locally built `dist/` directory.

## One-time setup

1. Create the dedicated public GitHub repository.
2. Enable branch protection and required CI checks.
3. Enable GitHub private vulnerability reporting.
4. Create protected GitHub environments named `testpypi` and `pypi`.
5. Configure matching Trusted Publishers on TestPyPI and PyPI.
6. Require manual approval for the `pypi` environment.
7. Configure GitHub Pages to use GitHub Actions.

## Release checklist

1. Confirm licence, authors, maintainers, and project URLs in `pyproject.toml`.
2. Update `CHANGELOG.md` and remove the `Unreleased` placeholder contents.
3. Set one new PEP 440 version in `src/mfdro/__init__.py`.
4. Run all commands in `CONTRIBUTING.md`.
5. Commit and merge through the protected default branch.
6. Run the TestPyPI workflow and install the uploaded package in a clean
   environment.
7. Create an annotated tag `vX.Y.Z` matching the package version.
8. Create a GitHub release from that tag.
9. Approve the protected PyPI environment after the release workflow validates
   the distributions.
10. Verify the PyPI page, installation command, documentation links, and
    provenance attestations.

Published versions and filenames cannot be overwritten. A defective release
must receive a new version; use PyPI yanking only when necessary.
