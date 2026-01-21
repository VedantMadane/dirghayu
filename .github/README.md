# GitHub Actions Configuration

## Workflows Configured

### 1. `ci.yml` - Continuous Integration
**Triggers:** Push to main/develop, Pull requests
**Runtime:** ~1.5 minutes (3 Python versions × 30 seconds)
**What it does:**
- Installs dependencies with uv (10x faster!)
- Lints code with ruff
- Runs tests with pytest
- Uploads coverage

### 2. `api-test.yml` - API Integration Tests
**Triggers:** Changes to API/model/data code
**Runtime:** ~1 minute
**What it does:**
- Starts FastAPI server
- Tests all endpoints
- Validates OpenAPI spec
- Ensures API functionality

### 3. `model-training.yml` - ML Model Training
**Triggers:** Manual or weekly schedule
**Runtime:** ~5 minutes (CPU), ~30 min (GPU if enabled)
**What it does:**
- Trains nutrient deficiency model
- Saves trained models as artifacts
- Can extend to GCP GPU training

### 4. `release.yml` - Release Management
**Triggers:** Git tags (v1.0.0, etc.)
**Runtime:** ~2 minutes
**What it does:**
- Builds Python package
- Creates GitHub release
- Can publish to PyPI (when ready)

### 5. `dependabot.yml` - Dependency Updates
**Schedule:** Weekly
**What it does:**
- Checks for outdated dependencies
- Opens PRs automatically
- Keeps project up-to-date

## Quick Setup

### 1. Push to GitHub
```bash
cd c:\Projects\open-source\dirghayu
git remote add origin https://github.com/YOUR_USERNAME/dirghayu.git
git push -u origin main
```

### 2. Enable Actions
- Go to repository → Settings → Actions
- Enable "Allow all actions and reusable workflows"

### 3. Add Secrets (if needed)
For GCP GPU training (optional):
- Settings → Secrets → Actions → New repository secret
- Name: `GCP_SA_KEY`
- Value: [GCP service account JSON]

### 4. Watch It Run
- Go to Actions tab
- See workflows running automatically!

## Cost Optimization

**With uv (already configured):**
- CI runtime: 1.5 min (vs 12 min with pip)
- **10.5 minutes saved per commit**
- **FREE if public repo anyway!**

## Badge for README

Add to your README.md:
```markdown
[![CI](https://github.com/YOUR_USERNAME/dirghayu/workflows/CI/badge.svg)](https://github.com/YOUR_USERNAME/dirghayu/actions)
```

## Questions?

See: `docs/github_actions_setup.md` for detailed documentation.
