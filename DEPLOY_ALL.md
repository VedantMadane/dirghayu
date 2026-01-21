# 🚀 Deploy Dirghayu to ALL Platforms

Complete guide to deploy your genomics platform to **5 platforms** simultaneously!

---

## 📋 Quick Overview

| # | Platform | Time | URL Format | Status |
|---|----------|------|------------|--------|
| 1️⃣ | **GitHub Pages** | 2 min | `vedantmadane.github.io/dirghayu` | ✅ DEPLOYED |
| 2️⃣ | **HuggingFace** | 5 min | `huggingface.co/spaces/vedant/dirghayu` | ⏳ In Progress |
| 3️⃣ | **Streamlit** | 3 min | `vedant-dirghayu.streamlit.app` | ⏸️ Ready |
| 4️⃣ | **Render** | 5 min | `dirghayu.onrender.com` | ⏸️ Ready |
| 5️⃣ | **Railway** | 4 min | `dirghayu.up.railway.app` | ⏸️ Ready |

**Total deployment time:** ~20 minutes for all platforms!

---

## 1️⃣ GitHub Pages (Static Demo) ✅ DEPLOYED

**Status:** ✅ Already Live  
**URL:** https://vedantmadane.github.io/dirghayu/  
**Best for:** Showcasing sample analysis reports

**What it shows:**
- Beautiful orange-themed UI
- Interactive risk cards
- Sample genetic insights
- Floating action menu (Share/Download/Print)

---

## 2️⃣ HuggingFace Spaces (Interactive Gradio) ⏳

**Status:** ⏳ In Progress  
**URL:** https://huggingface.co/spaces/vedant/dirghayu  
**Best for:** File upload & real-time analysis

### Setup Steps:
1. **Create Space:** https://huggingface.co/new-space
   - Name: `dirghayu`
   - SDK: **Gradio**
   - Hardware: **CPU basic (free)**
   - Visibility: **Public**

2. **Push code:**
   ```powershell
   cd c:\Projects\open-source\dirghayu
   git push hf main
   ```

3. **Wait 2-3 minutes** for build to complete

### Free Tier:
- ✅ **Unlimited** CPU basic instances
- ✅ Always-on (doesn't sleep)
- ✅ 16 GB persistent storage
- ⚠️ Slower than paid GPU instances

---

## 3️⃣ Streamlit Cloud ☁️

**URL:** https://vedant-dirghayu.streamlit.app  
**Best for:** Beautiful data apps with instant updates

### Setup Steps:
1. **Go to:** https://share.streamlit.io/

2. **Sign in** with GitHub

3. **Click:** "New app"

4. **Fill in:**
   - Repository: `VedantMadane/dirghayu`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - App URL: `vedant-dirghayu` (or any name)

5. **Click:** "Deploy!"

6. **Done!** App will be live in 2-3 minutes

### Free Tier:
- ✅ **Unlimited** public apps
- ✅ 1 GB RAM per app
- ✅ Always-on
- ✅ Auto-deploys on git push
- ⚠️ Community Cloud tier

### Files Created:
- ✅ `streamlit_app.py` - Full Streamlit app
- ✅ `requirements-streamlit.txt` - Dependencies

---

## 4️⃣ Render (FastAPI Production) 🔧

**URL:** https://dirghayu.onrender.com  
**Best for:** Production-ready REST API

### Setup Steps:
1. **Go to:** https://render.com

2. **Sign in** with GitHub

3. **Click:** "New" → "Web Service"

4. **Connect repository:** `VedantMadane/dirghayu`

5. **Settings:**
   - Name: `dirghayu`
   - Region: **Singapore** (closest to India)
   - Branch: `main`
   - Runtime: **Python 3**
   - Build Command: `pip install -r requirements-render.txt`
   - Start Command: `uvicorn src.api.server:app --host 0.0.0.0 --port $PORT`

6. **Click:** "Create Web Service"

7. **Wait 5-10 minutes** for first build

### Free Tier:
- ✅ **750 hours/month** (31.25 days)
- ✅ 512 MB RAM
- ✅ Auto-deploys on push
- ⚠️ Spins down after 15 min inactivity (cold start ~30s)
- ⚠️ Free plan expires after 90 days

### API Endpoints:
Once deployed, visit:
- 📖 API Docs: `https://dirghayu.onrender.com/docs`
- ❤️ Health Check: `https://dirghayu.onrender.com/health`
- 🧬 Analyze: `POST https://dirghayu.onrender.com/analyze`

### Files Created:
- ✅ `render.yaml` - Auto-configuration
- ✅ `requirements-render.txt` - Minimal dependencies

---

## 5️⃣ Railway 🚂

**URL:** https://dirghayu.up.railway.app  
**Best for:** Quick prototypes with database

### Setup Steps:
1. **Go to:** https://railway.app

2. **Sign in** with GitHub

3. **Click:** "New Project"

4. **Select:** "Deploy from GitHub repo"

5. **Choose:** `VedantMadane/dirghayu`

6. **Settings** (auto-detected from `railway.toml`):
   - Build: `pip install -r requirements-hf.txt`
   - Start: `python app_hf.py`

7. **Click:** "Deploy"

8. **Generate domain:**
   - Go to project settings
   - Click "Generate Domain"
   - Get: `dirghayu.up.railway.app`

### Free Tier:
- ✅ **$5 credit/month** (~100-140 hours)
- ✅ 8 GB RAM
- ✅ 100 GB bandwidth
- ⚠️ Sleeps when credit exhausted
- ⚠️ Credit resets monthly

### Files Created:
- ✅ `railway.json` - JSON config
- ✅ `railway.toml` - TOML config (preferred)

---

## 📊 Platform Comparison

| Feature | HuggingFace | Streamlit | Render | Railway | GitHub Pages |
|---------|-------------|-----------|--------|---------|--------------|
| **Type** | Gradio UI | Streamlit UI | FastAPI | Flexible | Static HTML |
| **Sleep?** | ❌ Never | ❌ Never | ✅ 15min | ✅ Credit | ❌ Never |
| **Build Time** | 2-3 min | 2-3 min | 5-10 min | 3-5 min | Instant |
| **Cold Start** | None | None | ~30s | ~10s | None |
| **File Upload** | ✅ Yes | ✅ Yes | ✅ API | ✅ Yes | ❌ No |
| **Database** | ❌ No | ❌ No | ✅ Add-on | ✅ Built-in | ❌ No |
| **Custom Domain** | ❌ No | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Best For** | ML demos | Data apps | APIs | Prototypes | Showcases |

---

## 🎯 Recommended Deployment Strategy

### For Testing (You!):
Deploy to **all 5 platforms** to test capabilities:

1. ✅ **GitHub Pages** - Already done! (Static demo)
2. ⏳ **HuggingFace** - Complete first (interactive demo)
3. ☁️ **Streamlit** - Deploy second (easiest UI)
4. 🔧 **Render** - Deploy third (production API)
5. 🚂 **Railway** - Deploy last (backup/alternative)

### For Production:
Pick **2 platforms**:
- **Primary:** HuggingFace or Streamlit (UI)
- **API:** Render (backend services)

---

## 🔄 Auto-Deployment Setup

All platforms support **auto-deploy on git push**!

### To enable:
1. **Push to GitHub** (main branch)
2. Platforms automatically detect changes
3. Rebuild and redeploy (2-10 min)

### Test auto-deploy:
```powershell
# Make a small change
echo "# Auto-deploy test" >> README.md
git add README.md
git commit -m "Test auto-deploy"
git push origin main

# Watch deployments across all platforms!
```

---

## 💰 Total Cost Analysis

| Platform | Monthly Cost | Hours/Month | Sufficient for Testing? |
|----------|--------------|-------------|------------------------|
| GitHub Pages | **FREE** | Unlimited | ✅ Perfect |
| HuggingFace | **FREE** | Unlimited | ✅ Perfect |
| Streamlit | **FREE** | Unlimited | ✅ Perfect |
| Render | **FREE** | 750 hours | ✅ Yes (31 days) |
| Railway | **FREE** | ~100-140 hrs | ✅ Yes (4-6 days) |
| **TOTAL** | **$0** | N/A | ✅ **Excellent for testing!** |

**Verdict:** You can deploy to **all 5 platforms for FREE** for testing! 🎉

---

## 📍 Your Deployment URLs

Once deployed, you'll have:

| Platform | Your URL |
|----------|----------|
| 🏠 Source | https://github.com/VedantMadane/dirghayu |
| 📄 Demo | https://vedantmadane.github.io/dirghayu/ |
| 🤗 App | https://huggingface.co/spaces/vedant/dirghayu |
| ☁️ App | https://vedant-dirghayu.streamlit.app |
| 🔧 API | https://dirghayu.onrender.com |
| 🚂 App | https://dirghayu.up.railway.app |

---

## 🚀 Next Steps

### Right Now:
1. **Finish HuggingFace:** Create space, push code
2. **Deploy Streamlit:** 3 clicks at share.streamlit.io
3. **Deploy Render:** Connect GitHub, configure
4. **Deploy Railway:** One-click deploy

### Want me to:
- ✅ Guide you through HuggingFace first?
- ✅ Deploy all platforms together?
- ✅ Focus on one platform?

**Let me know and we'll get all platforms live!** 🚀
