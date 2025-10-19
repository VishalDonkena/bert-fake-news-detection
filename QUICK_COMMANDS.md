# ⚡ Quick Commands Cheat Sheet

## 🚀 GitHub Upload Commands

```bash
# Navigate to project directory
cd /Users/vishaldonkena/Code/fake_news_detector_bert

# Check git status
git status

# Create GitHub repository (using GitHub CLI - optional)
gh repo create bert-fake-news-detection --public --description "🔍 BERT-based fake news detection with multiple training options and comprehensive evaluation tools"

# Or create manually at: https://github.com/new

# Connect to GitHub (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/bert-fake-news-detection.git
git branch -M main
git push -u origin main
```

## 🧪 Testing Commands

```bash
# Quick batch test
python3 test_model.py --quick

# Interactive testing (most fun)
python3 evaluate_model.py --model_path ./models_bert_minimal --interactive

# Single prediction
python3 evaluate_model.py --model_path ./models_bert_minimal --text "Your news article here"

# Test with custom input
python3 test_model.py --custom

# Compare different models
python3 test_model.py --model ./models_bert_minimal
python3 test_model.py --model ./models_bert_fast
```

## 🏋️‍♀️ Training Commands

```bash
# Quick training (1-2 minutes)
python3 train_bert_minimal.py

# Balanced training (5-10 minutes)
python3 train_bert_fast.py

# Production training with config
python3 train_bert_production.py --config config.json

# Create default config
python3 train_bert_production.py --create-config my_config.json

# Resume from checkpoint
python3 train_bert_production.py --resume ./output
```

## 📊 Evaluation Commands

```bash
# Comprehensive evaluation on test data
python3 evaluate_model.py --model_path ./models_bert_fast --test_data ./data/test.csv

# Generate detailed report
python3 evaluate_model.py --model_path ./models_bert_fast --test_data ./data/test.csv --output_dir ./evaluation_results

# Interactive testing session
python3 evaluate_model.py --model_path ./models_bert_fast --interactive
```

## 🔧 Setup Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Check available models
ls -la models_*

# View project structure
tree -L 2

# Check GPU availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🎯 Example Test Articles

### Real News Examples:
```bash
python3 evaluate_model.py --model_path ./models_bert_minimal --text "The Federal Reserve announced a 0.25% interest rate increase following their monthly meeting to address inflation concerns."

python3 evaluate_model.py --model_path ./models_bert_minimal --text "Local fire department responded to a house fire on Oak Street this morning, with no injuries reported."

python3 evaluate_model.py --model_path ./models_bert_minimal --text "Scientists at Stanford University published a new study on renewable energy efficiency in the journal Nature."
```

### Fake News Examples:
```bash
python3 evaluate_model.py --model_path ./models_bert_minimal --text "BREAKING: Aliens confirm they built the pyramids and are returning next week!"

python3 evaluate_model.py --model_path ./models_bert_minimal --text "You won't believe this ONE WEIRD TRICK that doctors HATE! Lose 50 pounds instantly!"

python3 evaluate_model.py --model_path ./models_bert_minimal --text "SHOCKING: Celebrity reveals they've been dead for 3 years but nobody noticed!"
```

## 📱 One-Liners for Quick Testing

```bash
# Test suspicious headline
python3 evaluate_model.py --model_path ./models_bert_minimal --text "Scientists discover chocolate is actually a vegetable!"

# Test normal news
python3 evaluate_model.py --model_path ./models_bert_minimal --text "Stock markets closed mixed today following economic data release"

# Batch test multiple articles
python3 test_model.py --samples

# Quick performance check
python3 test_model.py --quick
```

## 🚨 Troubleshooting Commands

```bash
# Check Python and package versions
python3 --version
pip list | grep torch
pip list | grep transformers

# Reinstall packages if needed
pip install --upgrade torch transformers

# Check model files
ls -la models_bert_*/
du -sh models_bert_*

# View training logs
tail -f training.log

# Clean up temporary files
rm -rf __pycache__ *.pyc
```

## 🔄 Git Commands for Updates

```bash
# Check what changed
git status

# Add new changes
git add .

# Commit changes
git commit -m "Add new feature: improved evaluation metrics"

# Push to GitHub
git push origin main

# Pull latest changes
git pull origin main

# View commit history
git log --oneline -10
```

## 🎯 Quick Project Demo

```bash
# Full demo sequence
echo "🚀 BERT Fake News Detection Demo"
echo "================================"

echo "1. Training a quick model..."
python3 train_bert_minimal.py

echo "2. Testing with sample articles..."
python3 test_model.py --quick

echo "3. Interactive testing (type your own articles)..."
python3 evaluate_model.py --model_path ./models_bert_minimal --interactive
```

## 📋 Pre-Upload Checklist

```bash
# Verify all files are ready
git status
git log --oneline -5

# Test core functionality
python3 test_model.py --quick

# Check README looks good
head -20 README.md

# Verify requirements
pip install -r requirements.txt

# Final git check
git remote -v
```

## 🎉 Success Verification

```bash
# After GitHub upload, verify:
# 1. Visit: https://github.com/YOUR_USERNAME/bert-fake-news-detection
# 2. Check all files are there
# 3. README displays correctly
# 4. Clone test:
git clone https://github.com/YOUR_USERNAME/bert-fake-news-detection.git test_clone
cd test_clone
python3 test_model.py --quick
```

---

**💡 Pro Tips:**
- Use `python3` instead of `python` for compatibility
- Always test commands in the correct directory
- Check `git status` before committing
- Use meaningful commit messages
- Test your uploaded repo by cloning it fresh

**🆘 Need Help?**
- Check `GITHUB_SETUP.md` for detailed instructions
- View `README.md` for comprehensive documentation
- Run `python3 script_name.py --help` for command options