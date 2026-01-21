# GitHub Actions Setup for Dirghayu

## Is GitHub Actions Free?

### YES for Public Repositories! 🎉

**Public repos (open-source projects):**
- ✅ **UNLIMITED** minutes
- ✅ **FREE** forever
- ✅ All features included
- ✅ Linux, Windows, macOS runners

**Private repos:**
| Plan | Free Minutes/Month | After Free Tier |
|------|-------------------|-----------------|
| Free | 2,000 minutes | $0.008/minute |
| Pro | 3,000 minutes | $0.008/minute |
| Team | 3,000 minutes | $0.008/minute |
| Enterprise | 50,000 minutes | $0.008/minute |

**Storage:**
- 500 MB free for all plans
- Artifacts retained for 90 days (configurable)

### Recommendation for Dirghayu
**Make the repository PUBLIC** → Get unlimited free CI/CD!

---

## What We've Configured

### 1. CI Workflow (`.github/workflows/ci.yml`)

**Runs on:**
- Every push to `main` or `develop`
- Every pull request

**What it does:**
- ✅ Tests on Python 3.10, 3.11, 3.12 (matrix)
- ✅ Installs deps with **uv** (~10 seconds vs 3 min with pip)
- ✅ Runs linting (ruff)
- ✅ Runs tests (pytest)
- ✅ Tests VCF parsing
- ✅ Uploads coverage to Codecov

**Time saved with uv:**
- Old (pip): ~4 minutes per job × 3 Python versions = 12 minutes
- New (uv): ~30 seconds per job × 3 = 1.5 minutes
- **Savings: 10.5 minutes per push!**

### 2. Model Training Workflow (`.github/workflows/model-training.yml`)

**Runs on:**
- Manual trigger (workflow_dispatch)
- Weekly schedule (Sundays)

**What it does:**
- ✅ Trains nutrient deficiency model
- ✅ Uploads trained model as artifact
- ✅ Can extend to GCP GPU training (commented out)

**Use case:**
- Retrain models on new data weekly
- Manual trigger when data updates

### 3. API Testing Workflow (`.github/workflows/api-test.yml`)

**Runs on:**
- Changes to `src/api/`, `src/models/`, `src/data/`

**What it does:**
- ✅ Starts FastAPI server
- ✅ Tests all endpoints
- ✅ Validates OpenAPI spec
- ✅ Uploads spec as artifact

### 4. Release Workflow (`.github/workflows/release.yml`)

**Runs on:**
- Git tags (`v1.0.0`, `v1.0.1`, etc.)

**What it does:**
- ✅ Builds Python package
- ✅ Creates GitHub release
- ✅ Attaches distribution files
- 🔒 PyPI upload (commented - add when ready)

### 5. Dependabot (`.github/dependabot.yml`)

**Automatically:**
- ✅ Checks for dependency updates weekly
- ✅ Opens PRs for outdated packages
- ✅ Updates GitHub Actions versions

---

## Speed Optimizations

### Using uv Instead of pip

**Traditional pip approach:**
```yaml
- name: Install dependencies
  run: pip install -r requirements.txt
  # Takes: 2-3 minutes
```

**Optimized uv approach:**
```yaml
- name: Install uv
  run: curl -LsSf https://astral.sh/uv/install.sh | sh

- name: Install dependencies  
  run: |
    export PATH="$HOME/.cargo/bin:$PATH"
    uv pip install --system -r requirements.txt
  # Takes: 10-15 seconds!
```

### Real Savings

**Per CI run:**
- 3 Python versions × 2.5 minutes saved = **7.5 minutes saved**

**Daily impact (10 commits/day):**
- 10 commits × 7.5 minutes = **75 minutes/day saved**

**Monthly impact:**
- 75 min/day × 20 working days = **1,500 minutes/month**
- At $0.008/min = **$12/month saved** (if private repo)
- For public repo = **FREE anyway**, but faster feedback!

---

## Usage Examples

### Trigger Manual Model Training

```bash
# Via GitHub CLI
gh workflow run model-training.yml -f model_type=nutrient

# Or via GitHub web UI:
# Actions tab → Train Models → Run workflow
```

### Check CI Status

```bash
# Check recent runs
gh run list

# Watch a running workflow
gh run watch

# View logs
gh run view <run-id> --log
```

### Download Trained Models

```bash
# Download artifacts from latest successful run
gh run download --name trained-models
```

---

## Cost Comparison: pip vs uv

### Monthly CI/CD Costs (Private Repo Scenario)

**Assumptions:**
- 50 commits/month
- 3 Python versions tested per commit
- 150 CI runs total

**With pip:**
- Time per run: 4 minutes
- Total: 150 × 4 = 600 minutes
- Within free tier if < 2,000 min
- Cost if over: $0.008 × (600 - 2000) = $0 (within free tier)

**With uv:**
- Time per run: 30 seconds = 0.5 minutes
- Total: 150 × 0.5 = 75 minutes
- **Massive headroom in free tier!**
- Faster developer feedback

### Time to Feedback

**Developer push to CI results:**

| Setup | Install Time | Test Time | Total Feedback |
|-------|--------------|-----------|----------------|
| pip | 3 min | 30 sec | 3.5 min |
| uv | 10 sec | 30 sec | 40 sec |

**uv gives 5x faster feedback!**

---

## Advanced: GPU Training on GCP (via GitHub Actions)

### Setup (One-time)

1. **Create GCP Service Account:**
```bash
gcloud iam service-accounts create dirghayu-ci \
  --display-name="Dirghayu CI/CD"

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:dirghayu-ci@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

2. **Add Secret to GitHub:**
```bash
# Get service account key
gcloud iam service-accounts keys create key.json \
  --iam-account=dirghayu-ci@PROJECT_ID.iam.gserviceaccount.com

# Add to GitHub repo secrets as GCP_SA_KEY
```

3. **Uncomment GPU training job in `model-training.yml`**

### Cost of GPU Training

**GCP Pricing (asia-southeast1-c):**
- n1-standard-4 + T4 GPU: ~$0.35/hour
- Training time: ~30 minutes
- Cost per run: ~$0.18

**When to use:**
- Large models (longevity predictor, disease risk models)
- Training on full GenomeIndia dataset
- Production model updates

**When NOT to use:**
- Small models (nutrient predictor) - CPU is fine
- Development/testing - use local machine

---

## Workflow Triggers Summary

| Workflow | Trigger | Frequency | Purpose |
|----------|---------|-----------|---------|
| CI | Push, PR | Every commit | Test code quality |
| API Test | Changes to API | As needed | Ensure API works |
| Model Training | Manual, Schedule | Weekly | Retrain models |
| Release | Git tag | On version tag | Publish releases |
| Dependabot | Schedule | Weekly | Update dependencies |

---

## Best Practices

### 1. Cache Dependencies
```yaml
- uses: actions/cache@v3
  with:
    path: ~/.cache/uv
    key: ${{ runner.os }}-uv-${{ hashFiles('requirements.txt') }}
```

### 2. Fail Fast
```yaml
strategy:
  fail-fast: false  # Test all Python versions even if one fails
```

### 3. Concurrency Limits
```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true  # Cancel old runs when new push
```

### 4. Conditional Steps
```yaml
- name: Upload to PyPI
  if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/')
```

---

## Monitoring CI/CD Usage

### Check Minutes Used (Private Repos)

```bash
# Via GitHub CLI
gh api /user/settings/billing/actions

# Or check in GitHub UI:
# Settings → Billing → Usage this month
```

### Current Setup Estimate

**For Dirghayu (if private repo):**
- 50 commits/month
- 3 Python versions × 0.5 min = 1.5 min per commit
- Total: 75 minutes/month
- **Well within 2,000 min free tier!**

---

## Adding More Workflows

### Example: Daily Data Sync
```yaml
name: Sync GenomeIndia Data

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Download latest GenomeIndia data
        run: python scripts/sync_genome_india.py
```

### Example: Security Scanning
```yaml
name: Security Scan

on:
  schedule:
    - cron: '0 0 * * 1'  # Weekly on Monday

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Bandit security scanner
        run: |
          pip install bandit
          bandit -r src/
```

---

## Alternatives to GitHub Actions

If you need more compute:

| Platform | Free Tier | Best For |
|----------|-----------|----------|
| **GitHub Actions** | Unlimited (public) | General CI/CD |
| GitLab CI | 400 min/month | GitLab users |
| CircleCI | 6,000 min/month | Docker workflows |
| Travis CI | 10,000 credits | Legacy projects |
| Azure Pipelines | 1,800 min/month | Microsoft stack |

**For Dirghayu: Stick with GitHub Actions** - It's free for open-source and well-integrated.

---

## Summary

**GitHub Actions for Dirghayu:**
- ✅ **FREE** (if public repo)
- ✅ **Fast** (uv makes it 10x faster)
- ✅ **Comprehensive** (CI, API tests, model training, releases)
- ✅ **Low maintenance** (Dependabot auto-updates)

**Estimated monthly usage (if private):**
- ~75 minutes/month
- **FREE** (within 2,000 min tier)

**Make repo public → UNLIMITED FREE forever!**
