# 🔍 BERT Fake News Detection

A comprehensive machine learning project for detecting fake news using fine-tuned BERT models. This repository provides multiple training options, from quick local testing to production-ready cloud training with full evaluation capabilities.

## 🚀 Quick Start

### Option 1: Minimal Training (1-2 minutes)
Perfect for testing and quick validation:
```bash
python train_bert_minimal.py
```

### Option 2: Fast Training (5-10 minutes)
Good balance between speed and performance:
```bash
python train_bert_fast.py
```

### Option 3: Google Colab Training (Free GPU)
Upload `BERT_Training_Colab.ipynb` to Google Colab for full training with GPU acceleration.

### Option 4: Production Training
For serious projects with checkpointing and monitoring:
```bash
python train_bert_production.py --config config.json
```

## 📊 Results Summary

| Model | Training Time | Validation Accuracy | Dataset Size |
|-------|---------------|-------------------|--------------|
| Minimal | ~1 minute | 100% | 100 samples |
| Fast | ~5 minutes | 99.5% | 1,000 samples |
| Full | ~30 minutes | 95-98% | Full dataset |

## 🔧 Installation

```bash
# Install required packages
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn

# Or install from requirements
pip install -r requirements.txt
```

## 📁 Project Structure

```
fake_news_detector_bert/
├── README.md                     # This file
├── requirements.txt              # Dependencies
├── config.json                   # Production training config
├── BERT_Training_Colab.ipynb     # Google Colab notebook
│
├── Training Scripts:
│   ├── train_bert.py             # Original training script
│   ├── train_bert_minimal.py     # Quick testing (1-2 min)
│   ├── train_bert_fast.py        # Balanced training (5-10 min)
│   └── train_bert_production.py  # Full production pipeline
│
├── Evaluation:
│   └── evaluate_model.py         # Comprehensive evaluation tools
│
└── Models (after training):
    ├── models_bert_minimal/       # Minimal model output
    ├── models_bert_fast/         # Fast model output
    └── output/                   # Production model output
```

## 🎯 Usage Guide

### 1. Quick Testing (Recommended First Step)

Start with the minimal version to ensure everything works:

```bash
python train_bert_minimal.py
```

**Features:**
- ✅ Uses only 100 samples
- ✅ DistilBERT (faster than BERT)
- ✅ Completes in 1-2 minutes
- ✅ Perfect for debugging

### 2. Balanced Training

Once you've verified the setup:

```bash
python train_bert_fast.py
```

**Features:**
- ✅ Uses 1,000 samples
- ✅ Full BERT model
- ✅ Good performance in 5-10 minutes
- ✅ Suitable for most use cases

### 3. Cloud Training with Google Colab

For full dataset training with free GPU:

1. Upload `BERT_Training_Colab.ipynb` to [Google Colab](https://colab.research.google.com)
2. Enable GPU runtime: Runtime → Change runtime type → GPU
3. Upload your `news.csv` data file
4. Run all cells

**Features:**
- ✅ Free GPU acceleration
- ✅ Full dataset training
- ✅ Interactive visualizations
- ✅ Model download capability

### 4. Production Training

For serious deployments with monitoring:

```bash
# Create default config
python train_bert_production.py --create-config config.json

# Start training
python train_bert_production.py --config config.json

# Resume from checkpoint
python train_bert_production.py --resume ./output
```

**Features:**
- ✅ Automatic checkpointing
- ✅ TensorBoard monitoring
- ✅ Early stopping
- ✅ Multi-GPU support
- ✅ Data augmentation
- ✅ Advanced scheduling

## 🧪 Model Evaluation

After training, evaluate your models:

```bash
# Evaluate on test data
python evaluate_model.py --model_path ./models_bert_fast --test_data ./data/test.csv

# Interactive testing
python evaluate_model.py --model_path ./models_bert_fast --interactive

# Single prediction
python evaluate_model.py --model_path ./models_bert_fast --text "Your news article here"
```

**Evaluation Features:**
- ✅ Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
- ✅ Confusion matrix visualization
- ✅ Interactive testing interface
- ✅ Batch prediction on new data
- ✅ Detailed performance reports

## 📊 Configuration Options

### Training Parameters

| Parameter | Minimal | Fast | Production |
|-----------|---------|------|------------|
| Model | DistilBERT | BERT-base | BERT-base |
| Max Length | 64 | 128 | 512 |
| Batch Size | 4 | 8 | 16 |
| Epochs | 1 | 1 | 3 |
| Samples | 100 | 1,000 | Full |

### Custom Configuration

For production training, modify `config.json`:

```json
{
  "data_path": "path/to/your/data.csv",
  "model_name": "bert-base-uncased",
  "max_length": 512,
  "batch_size": 16,
  "epochs": 3,
  "learning_rate": 2e-5,
  "early_stopping": true,
  "data_augmentation": false
}
```

## 📈 Performance Tips

### For Faster Training:
1. Use **DistilBERT** instead of BERT (`distilbert-base-uncased`)
2. Reduce **max_length** (256 instead of 512)
3. Increase **batch_size** if you have enough memory
4. Use **data sampling** for initial experiments

### For Better Performance:
1. Use **full dataset** with proper train/val/test splits
2. Enable **data_augmentation**
3. Try **different learning rates** (1e-5, 2e-5, 3e-5, 5e-5)
4. Use **early stopping** to prevent overfitting

### For Production:
1. Enable **checkpointing** and **monitoring**
2. Use **multiple epochs** with **early stopping**
3. Implement **proper logging**
4. Set up **model validation** pipelines

## 🔍 Troubleshooting

### Common Issues:

**"Out of Memory" Error:**
```bash
# Reduce batch size
BATCH_SIZE = 4  # instead of 16

# Reduce max length
MAX_LENGTH = 256  # instead of 512
```

**Training Too Slow:**
```bash
# Use the minimal version first
python train_bert_minimal.py

# Or switch to DistilBERT
MODEL_NAME = "distilbert-base-uncased"
```

**CUDA Not Available:**
- The scripts automatically fall back to CPU
- Training will be slower but still functional
- Consider using Google Colab for free GPU

**Model Loading Issues:**
```bash
# Check if model files exist
ls models_bert_fast/

# Verify the model path
python evaluate_model.py --model_path ./models_bert_fast --interactive
```

## 📚 Data Format

Your CSV file should have these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `text` | News article content | "Scientists discover new planet..." |
| `label` | 0 = Real News, 1 = Fake News | 0 or 1 |

Example CSV:
```csv
text,label
"Scientists have discovered a new planet that could support life.",0
"BREAKING: Aliens have landed in New York City!",1
```

## 🎨 Visualization Features

The training scripts provide:
- ✅ **Real-time training progress**
- ✅ **Loss and accuracy plots**
- ✅ **Confusion matrices**
- ✅ **ROC curves**
- ✅ **TensorBoard integration** (production mode)

## 🚀 Deployment Ready

Trained models can be used for:
- ✅ **Batch prediction** on new articles
- ✅ **Real-time API** endpoints
- ✅ **Web applications**
- ✅ **Mobile apps** (with model compression)

Example deployment code:
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load trained model
model = BertForSequenceClassification.from_pretrained('./models_bert_fast')
tokenizer = BertTokenizer.from_pretrained('./models_bert_fast')

def predict_news(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return "Fake" if prediction[0][1] > 0.5 else "Real"
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face Transformers** for the BERT implementation
- **Google** for the original BERT paper
- **PyTorch** team for the deep learning framework
- **scikit-learn** for evaluation metrics

## 📞 Support

- 📧 **Issues**: Open a GitHub issue for bugs or questions
- 💬 **Discussions**: Use GitHub Discussions for general questions
- 📖 **Documentation**: Check this README and code comments

---

**Happy fake news hunting! 🔍🗞️**