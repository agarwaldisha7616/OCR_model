# Handwriting Recognition - Quick Start Guide

## Installation & Setup

```bash
# 1. Navigate to project directory
cd "d:\projects\Handwriting Recognition"

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Running the Application

```bash
# Start the Streamlit dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501`

## First Time Setup

1. **Install Python 3.8+** if not already installed
2. **Install Git** (optional, for version control)
3. **GPU Setup** (optional, for faster inference):
   - Install NVIDIA CUDA toolkit
   - Install cuDNN
   - Update requirements: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

## Usage Steps

### 1. Upload a Document
- Click the file uploader in the main area
- Select a scanned handwritten document (JPG, PNG, BMP, TIFF)

### 2. Configure Settings
- In the sidebar, select your preferred OCR model:
  - **TrOCR**: Fast, optimized for handwritten text
  - **DocTR**: Comprehensive document analysis
- Optionally enable fine-tuned model if available

### 3. Extract Text
- View the original and preprocessed images side-by-side
- Click the "🚀 Extract Text" button
- Wait for processing to complete

### 4. Review Results
- Copy extracted text using the "📋 Copy" button
- Download as .txt file
- View performance metrics (inference time, word count, etc.)

## Training a Custom Model (Advanced)

To fine-tune TrOCR on your own data:

```bash
python train_trocr.py
```

This trains on the imgur5k_words dataset and saves to `trocr-finetuned/`

## File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit dashboard application |
| `ocr_models.py` | TrOCR and DocTR model wrappers |
| `preprocessing.py` | Image preprocessing functions |
| `train_trocr.py` | Training script for custom TrOCR models |
| `requirements.txt` | Python package dependencies |
| `README.md` | Full documentation |

## Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"
- Close other applications
- Use CPU mode (automatic fallback)
- Reduce batch processing size

### Poor text extraction quality
1. Ensure good image quality (300+ DPI)
2. Check contrast and lighting
3. Try alternative model selection
4. Enable fine-tuned model for handwritten text

### Model download fails
```bash
# Clear cache and retry
pip cache purge
pip install -r requirements.txt
```

## Common Commands

```bash
# Update packages
pip install -r requirements.txt --upgrade

# Run with custom port
streamlit run app.py --server.port=8080

# Run without opening browser
streamlit run app.py --logger.level=debug

# Kill stuck Streamlit process
taskkill /F /IM python.exe  # Windows
kill $(lsof -ti :8501)     # Linux/Mac
```

## Next Steps

1. **Try with sample documents** - Test with various image qualities
2. **Fine-tune model** - Run `train_trocr.py` with your own data
3. **Integrate DocTR** - Install `python-doctr` for alternative model
4. **Deploy** - Use Streamlit Cloud or Docker for production

## Resources

- 📚 [Streamlit Docs](https://docs.streamlit.io)
- 🤗 [Hugging Face Models](https://huggingface.co)
- 🎓 [PyTorch Tutorials](https://pytorch.org/tutorials)
- 🔍 [OpenCV Documentation](https://docs.opencv.org)

---

**Ready to digitize documents?** Start by running:
```bash
streamlit run app.py
```
