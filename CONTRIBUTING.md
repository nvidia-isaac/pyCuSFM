# PyCuSFM Contribution Rules

We welcome contributions to the PyCuSFM project! This document provides guidelines for contributing to ensure a smooth collaboration process.

## License Agreement

Any contribution that you make to this repository will be under the Apache 2 License, as dictated by that [license](http://www.apache.org/licenses/LICENSE-2.0.html).

## Issue Tracking

* All enhancement, bugfix, or change requests must begin with the creation of a [PyCuSFM Issue Request](https://github.com/nvidia-isaac/PyCuSFM/issues).
  * The issue request must be reviewed by PyCuSFM engineers and approved prior to code review.

## Coding Guidelines

- All source code contributions must strictly adhere to the existing code style and conventions.

- In addition, please follow the existing conventions in the relevant file, submodule, module, and project when you add new code or when you extend/fix existing functionality.

- To maintain consistency in code formatting and style, you should run the appropriate formatters on the modified sources:

- Avoid introducing unnecessary complexity into existing code so that maintainability and readability are preserved.

- Try to keep pull requests (PRs) as concise as possible:
  - Avoid committing commented-out code.
  - Wherever possible, each PR should address a single concern. If there are several otherwise-unrelated things that should be fixed to reach a desired endpoint, our recommendation is to open several PRs and indicate the dependencies in the description. The more complex the changes are in a single PR, the more time it will take to review those changes.

- Write commit titles using imperative mood and [these rules](https://chris.beams.io/posts/git-commit/), and reference the Issue number corresponding to the PR. Following is the recommended format for commit texts:
```
#<Issue Number> - <Commit Title>

<Commit Body>
```

- Ensure that the build log is clean, meaning no warnings or errors should be present.

- Ensure that all tests pass prior to submitting your code. The CI/CD pipeline will automatically run tests using the sample data in `data/r2b_galileo`.

- All OSS components must contain accompanying documentation (READMEs) describing the functionality, dependencies, and known issues.

- All OSS components must have an accompanying test.
  - If introducing a new component, provide a test to verify the functionality.
  - Tests should be compatible with the CI/CD pipeline which uses TensorRT 24.12 environment.
  - Functional tests should use the provided sample data or create minimal test datasets.

- Make sure that you can contribute your work to open source (no license and/or patent conflict is introduced by your code). You will need to [`sign`](#signing-your-work) your commit.

- Thanks in advance for your patience as we review your contributions; we do appreciate them!

## Development Setup

If you plan to contribute to this project, please set up the development environment with code quality tools.

### Install Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality and consistency. After cloning the repository, install the hooks:

```bash
# Install pre-commit (if not already installed)
pip install pre-commit

# Install the git hook scripts
pre-commit install --install-hooks
```

The pre-commit configuration includes:
- **Code formatting**: Automatic Python code formatting with yapf (PEP8 style, 79 character line limit)
- **Trailing whitespace**: Removes trailing whitespace from files
- **End-of-file fixer**: Ensures files end with a newline
- **YAML validation**: Checks YAML file syntax
- **Large file detection**: Prevents accidentally committing large files

### Manual Pre-commit Run

You can also run pre-commit manually on all files:

```bash
# Run on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files path/to/file.py
```

### Code Formatting Configuration

The yapf configuration follows these settings:
- Based on PEP8 style
- 79 character column limit
- Split arguments when comma terminated
- Split before first argument and expressions after opening parentheses

You can find the complete yapf configuration in `.style.yapf`.

## Pull Requests

Developer workflow for code contributions is as follows:

1. Developers must first [fork](https://help.github.com/en/articles/fork-a-repo) the [upstream](https://github.com/nvidia-isaac/PyCuSFM) PyCuSFM repository.

2. Git clone the forked repository and push changes to the personal fork.

   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_FORK.git pycusfm
   cd pycusfm
   # Checkout the targeted branch and commit changes
   # Push the commits to a branch on the fork (remote).
   git push -u origin <local-branch>:<remote-branch>
   ```

3. Once the code changes are staged on the fork and ready for review, a [Pull Request](https://help.github.com/en/articles/about-pull-requests) (PR) can be [requested](https://help.github.com/en/articles/creating-a-pull-request) to merge the changes from a branch of the fork into a selected branch of upstream.
   * Exercise caution when selecting the source and target branches for the PR.
   * Creation of a PR kicks off the code review process.
   * At least one PyCuSFM engineer will be assigned for the review.
   * While under review, mark your PRs as work-in-progress by prefixing the PR title with [WIP].

4. The PR will be accepted and the corresponding issue closed only after adequate testing has been completed, manually, by the developer and/or PyCuSFM engineer reviewing the code.

## Development Environment

Please refer to the [README.md](README.md) for detailed installation instructions including:
- Docker environment setup
- Build instructions

## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

## Developer Certificate of Origin (DCO)

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```

The full text of the DCO can be found at: https://developercertificate.org/

## Questions?

If you have questions about contributing, please:
1. Check existing documentation
2. Search through existing issues at https://github.com/nvidia-isaac/PyCuSFM/issues
3. Create a new issue with the "question" label

Thank you for contributing to PyCuSFM!
