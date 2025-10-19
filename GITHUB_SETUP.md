# 🚀 GitHub Repository Setup Guide

This guide will help you upload your BERT Fake News Detection project to GitHub.

## 📋 Prerequisites

Before starting, make sure you have:
- ✅ A GitHub account ([sign up here](https://github.com))
- ✅ Git installed on your computer ([download here](https://git-scm.com/))
- ✅ Your BERT project files ready

## 🔧 Step-by-Step Setup

### 1. Initialize Git Repository

Open terminal/command prompt in your project directory and run:

```bash
cd /Users/vishaldonkena/Code/fake_news_detector_bert
git init
```

### 2. Configure Git (if first time)

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 3. Add Files to Git

```bash
# Add all files except those in .gitignore
git add .

# Check what files are staged
git status
```

### 4. Create Initial Commit

```bash
git commit -m "Initial commit: BERT Fake News Detection project

- Multiple training options (minimal, fast, production)
- Google Colab notebook for GPU training
- Comprehensive evaluation tools
- Interactive testing interface
- Complete documentation and examples"
```

### 5. Create GitHub Repository

#### Option A: Using GitHub Web Interface (Recommended)

1. Go to [GitHub](https://github.com)
2. Click the **"+" icon** → **"New repository"**
3. Fill out the form:
   - **Repository name**: `bert-fake-news-detection`
   - **Description**: `🔍 BERT-based fake news detection with multiple training options and comprehensive evaluation tools`
   - **Visibility**: Choose Public or Private
   - ⚠️ **DON'T** initialize with README (we already have one)
4. Click **"Create repository"**

#### Option B: Using GitHub CLI (Alternative)

```bash
# Install GitHub CLI first: https://cli.github.com/
gh repo create bert-fake-news-detection --public --description "🔍 BERT-based fake news detection with multiple training options and comprehensive evaluation tools"
```

### 6. Connect Local Repository to GitHub

Replace `YOUR_USERNAME` with your actual GitHub username:

```bash
git remote add origin https://github.com/YOUR_USERNAME/bert-fake-news-detection.git
git branch -M main
git push -u origin main
```

### 7. Verify Upload

1. Go to your GitHub repository: `https://github.com/YOUR_USERNAME/bert-fake-news-detection`
2. You should see all your files uploaded
3. The README.md should display beautifully with all the documentation

## 📁 What Gets Uploaded

✅ **Included Files:**
- All Python scripts (training, evaluation, testing)
- Documentation (README.md, guides)
- Configuration files
- Requirements.txt
- Google Colab notebook
- Example scripts

❌ **Excluded Files (via .gitignore):**
- Trained model files (too large)
- Data files (privacy/size)
- Cache and temporary files
- Log files
- Virtual environments

## 🔄 Future Updates

When you make changes to your project:

```bash
# Check what changed
git status

# Add changes
git add .

# Commit changes
git commit -m "Add feature: new evaluation metrics"

# Push to GitHub
git push origin main
```

## 🎯 Repository Features to Add

After uploading, consider adding these GitHub features:

### 1. Repository Topics
Go to your repo → About section → Add topics:
- `machine-learning`
- `nlp`
- `bert`
- `fake-news-detection`
- `pytorch`
- `transformers`
- `python`

### 2. GitHub Pages (Optional)
Enable GitHub Pages to host your documentation:
1. Go to Settings → Pages
2. Source: Deploy from branch → `main` → `/ (root)`

### 3. Issues and Discussions
Enable Issues for bug reports and feature requests

### 4. GitHub Actions (Advanced)
Add automated testing when you push code

## 🔧 Troubleshooting

### Problem: "Repository already exists"
**Solution:** Choose a different name like `bert-fake-news-detector-v2`

### Problem: Files too large
**Solution:** 
```bash
# Remove large files
git rm --cached large_file.bin
git commit -m "Remove large file"
```

### Problem: Authentication failed
**Solutions:**
1. Use Personal Access Token instead of password
2. Set up SSH keys
3. Use GitHub Desktop app

### Problem: Can't push to GitHub
**Solutions:**
```bash
# Check remote URL
git remote -v

# Fix remote URL if needed
git remote set-url origin https://github.com/YOUR_USERNAME/bert-fake-news-detection.git
```

## 📞 Need Help?

- 📚 [Git Handbook](https://guides.github.com/introduction/git-handbook/)
- 🎥 [GitHub Video Guides](https://www.youtube.com/githubguides)
- 💬 [GitHub Community](https://github.community/)

## 🎉 Success!

Once uploaded, your repository URL will be:
`https://github.com/YOUR_USERNAME/bert-fake-news-detection`

Share this link to showcase your amazing BERT fake news detection project! 🚀

---

**Pro Tips:**
- ⭐ Star your own repository to bookmark it
- 📝 Create a good repository description
- 🏷️ Add relevant topics for discoverability
- 📋 Consider adding a LICENSE file
- 🎯 Add screenshots to your README for visual appeal