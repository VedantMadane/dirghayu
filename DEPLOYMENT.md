# Dirghayu Deployment Guide

## 📍 Current Status
- ✅ **Source Code:** https://github.com/VedantMadane/dirghayu
- ❌ **Live Demo:** Not deployed yet

---

## 🚀 Deployment Options

### Option 1: HuggingFace Spaces (RECOMMENDED) 🤗

**Best for:** Gradio web app with file upload  
**Cost:** FREE  
**URL:** `https://huggingface.co/spaces/YOUR_USERNAME/dirghayu`

#### Steps:
1. **Create HuggingFace Account:** https://huggingface.co/join
2. **Create New Space:**
   - Go to: https://huggingface.co/new-space
   - Name: `dirghayu`
   - SDK: Gradio
   - Visibility: Public
   
3. **Push Code:**
```bash
cd c:\Projects\open-source\dirghayu

# Add HuggingFace remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/dirghayu

# Copy simplified app
cp app_hf.py app.py
cp requirements-hf.txt requirements.txt

# Commit and push
git add app.py requirements.txt
git commit -m "Add HuggingFace deployment"
git push hf main
```

4. **Done!** Your app will be live at:
   `https://huggingface.co/spaces/YOUR_USERNAME/dirghayu`

---

### Option 2: Streamlit Cloud ☁️

**Best for:** Simple data apps  
**Cost:** FREE  
**URL:** `https://YOUR_USERNAME-dirghayu.streamlit.app`

#### Create Streamlit App:
```python
# streamlit_app.py
import streamlit as st
from data import parse_vcf_file

st.title("🧬 Dirghayu")
st.caption("India-First Longevity Genomics")

uploaded_file = st.file_uploader("Upload VCF", type=['vcf'])

if uploaded_file:
    variants_df = parse_vcf_file(uploaded_file)
    st.success(f"✅ {len(variants_df)} variants analyzed")
    st.dataframe(variants_df)
```

#### Deploy:
1. Go to: https://share.streamlit.io/
2. Connect GitHub repo: `VedantMadane/dirghayu`
3. Main file: `streamlit_app.py`
4. Deploy!

---

### Option 3: Render (FastAPI) 🔧

**Best for:** Production API server  
**Cost:** FREE tier available  
**URL:** `https://dirghayu.onrender.com`

#### Steps:
1. Create `render.yaml`:
```yaml
services:
  - type: web
    name: dirghayu-api
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "uvicorn src.api.server:app --host 0.0.0.0 --port $PORT"
    envVars:
      - key: PYTHON_VERSION
        value: 3.11
```

2. Deploy:
   - Go to: https://render.com
   - New → Web Service
   - Connect repo: `VedantMadane/dirghayu`
   - Auto-deploy!

---

### Option 4: GitHub Pages (Static Reports) 📄

**Best for:** Static HTML demo reports  
**Cost:** FREE  
**URL:** `https://VedantMadane.github.io/dirghayu/`

#### Steps:
```bash
# Generate sample report
python web_demo.py data/clinvar_sample.vcf

# Rename for GitHub Pages
mv dirghayu_report.html index.html

# Create docs folder
mkdir docs
mv index.html docs/

# Push to GitHub
git add docs/
git commit -m "Add demo report for GitHub Pages"
git push origin main

# Enable GitHub Pages
gh repo edit --enable-pages --pages-branch main --pages-path docs
```

**Live Demo:** https://VedantMadane.github.io/dirghayu/

---

### Option 5: Railway 🚂

**Best for:** Full-stack deployment  
**Cost:** $5/month credit FREE  
**URL:** `https://dirghayu.up.railway.app`

#### Steps:
1. Go to: https://railway.app
2. New Project → Deploy from GitHub
3. Select: `VedantMadane/dirghayu`
4. Add start command: `python app.py`
5. Deploy!

---

## 🎯 Recommended Setup

### For Maximum Reach:

1. **HuggingFace Spaces** - Interactive web app
   - URL: `https://huggingface.co/spaces/YOU/dirghayu`
   - Users can upload VCF files

2. **GitHub Pages** - Static demo
   - URL: `https://VedantMadane.github.io/dirghayu/`
   - Show sample analysis report

3. **FastAPI on Render** - Production API
   - URL: `https://dirghayu.onrender.com/docs`
   - OpenAPI documentation

---

## 📊 Cost Comparison

| Platform | Free Tier | Best For | URL Format |
|----------|-----------|----------|------------|
| **HuggingFace** | ✅ Unlimited | Gradio apps | `hf.co/spaces/USER/APP` |
| **Streamlit** | ✅ Unlimited | Data apps | `USER-APP.streamlit.app` |
| **GitHub Pages** | ✅ Unlimited | Static sites | `USER.github.io/REPO` |
| **Render** | ✅ 750 hrs/mo | APIs | `APP.onrender.com` |
| **Railway** | ✅ $5 credit | Full-stack | `APP.railway.app` |
| **Vercel** | ✅ Unlimited | Static/Next.js | `APP.vercel.app` |

---

## 🚀 Quick Deploy Commands

### Deploy to HuggingFace:
```bash
cd c:\Projects\open-source\dirghayu
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/dirghayu
git push hf main
```

### Deploy GitHub Pages:
```bash
python web_demo.py data/clinvar_sample.vcf
mkdir docs
mv dirghayu_report.html docs/index.html
git add docs/ && git commit -m "Add demo" && git push
gh repo edit --enable-pages --pages-branch main --pages-path docs
```

### Deploy Streamlit:
```bash
# Just connect your GitHub repo at https://share.streamlit.io/
# No commands needed - click deploy!
```

---

## 🌐 Expected URLs After Deployment

Once deployed, your project will be available at:

- 🤗 **HuggingFace:** `https://huggingface.co/spaces/YOUR_USERNAME/dirghayu`
- 📄 **GitHub Pages:** `https://VedantMadane.github.io/dirghayu/`
- ☁️ **Streamlit:** `https://YOUR_USERNAME-dirghayu.streamlit.app`
- 🔧 **Render API:** `https://dirghayu.onrender.com`

---

## 🎯 Next Steps

### 1. Choose Your Platform
Pick from: HuggingFace, Streamlit, GitHub Pages, or Render

### 2. Follow the Steps Above
Each platform has simple deployment instructions

### 3. Share Your Live Demo
Once deployed, share the URL!

---

## 📝 Notes

- **HuggingFace Spaces** - Best for interactive demos
- **GitHub Pages** - Best for showcasing results
- **Streamlit Cloud** - Easiest to deploy
- **Render/Railway** - Best for production APIs

---

## 🆘 Need Help?

1. HuggingFace Docs: https://huggingface.co/docs/hub/spaces
2. Streamlit Docs: https://docs.streamlit.io/deploy
3. Render Docs: https://render.com/docs
4. GitHub Pages: https://pages.github.com/

---

**Ready to deploy? Let's get your genomics platform live!** 🚀
