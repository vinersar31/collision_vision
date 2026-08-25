✨ Add GitHub Actions release workflow

🎯 **What:** Created a new GitHub Actions workflow (`.github/workflows/release.yml`) for creating release builds and tags.
✅ **Why:** To automate the process of creating release branches, packaging the application for deployment (including source code and excluding unnecessary development artifacts), and publishing a GitHub Release with the corresponding version tag.
🛠️ **How:**
- The workflow is manually triggered (`workflow_dispatch`) with a required `version` input.
- It checks out the `main` branch.
- It creates and pushes a new branch named `release/<version>`.
- It creates a clean `release-package.zip` archive with the source code, `Dockerfile`, `requirements.txt`, etc., while excluding `.git`, `.github`, `tests`, `website`, cached files, and training outputs/data.
- It utilizes the GitHub CLI (`gh release create`) to draft a new release using the zip artifact.
