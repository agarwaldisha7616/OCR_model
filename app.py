import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from preprocessing import preprocess_image
from ocr_models import load_ocr_model, predict_with_fallback

# Load environment variables from .env file
from pathlib import Path
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Page configuration
st.set_page_config(
    page_title="Historical Document Digitization Pipeline",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .stMetric label {
        color: #1f77b4 !important;
        font-weight: 600;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #2c3e50 !important;
        font-size: 2rem;
        font-weight: bold;
    }
    .success-box {
        background-color: #d1e7dd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #198754;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📄 Historical Document Digitization Pipeline")
st.markdown("Convert handwritten documents into digital text with advanced OCR technology")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

model_type = st.sidebar.selectbox(
    "Select OCR Model",
    options=["TrOCR", "DocTR"],
    help="TrOCR: Fast handwritten text recognition\nDocTR: Comprehensive document understanding"
)

use_gemini_fallback = st.sidebar.checkbox(
    "Use Gemini as Fallback",
    value=True,
    help="Automatically use Gemini AI if the primary model fails"
)

use_finetuned = st.sidebar.checkbox(
    "Use Fine-tuned TrOCR Model",
    value=False,
    disabled=(model_type != "TrOCR"),
    help="Use the fine-tuned model trained on handwritten text"
)

# Local model path input
local_model_path = None
if model_type == "TrOCR":
    use_local_model = st.sidebar.checkbox(
        "Use Local Model (.safetensors)",
        value=False,
        help="Load model from a local directory containing .safetensors files"
    )
    
    if use_local_model:
        local_model_path = st.sidebar.text_input(
            "Model Path",
            placeholder="e.g., D:/models/trocr-model",
            help="Enter the full path to your model directory or .safetensors file"
        )
        
        if local_model_path:
            import os
            if os.path.exists(local_model_path):
                st.sidebar.success(f"✅ Path found: {os.path.basename(local_model_path)}")
            else:
                st.sidebar.error(f"❌ Path not found!")
                local_model_path = None

st.sidebar.markdown("---")
st.sidebar.markdown("""
### How It Works:
1. **Upload** a noisy handwritten document image
2. **Preprocess** the image for optimal OCR
3. **Extract** text using the selected model
4. **Review** results and metrics

### Supported Formats:
- JPG, PNG, BMP, TIFF
- Max size: 200MB
""")

# Initialize session state for model caching
if "current_model" not in st.session_state:
    st.session_state.current_model = None
    st.session_state.current_model_type = None
    st.session_state.current_finetuned = None
    st.session_state.current_local_path = None

# Load model with caching
def load_model_cached(model_type, use_finetuned, local_model_path=None):
    # Check if we need to reload
    if (st.session_state.current_model is None or 
        st.session_state.current_model_type != model_type or
        st.session_state.current_finetuned != use_finetuned or
        st.session_state.current_local_path != local_model_path):
        
        with st.spinner(f"Loading {model_type} model..."):
            try:
                st.session_state.current_model = load_ocr_model(model_type, use_finetuned, local_model_path)
                st.session_state.current_model_type = model_type
                st.session_state.current_finetuned = use_finetuned
                st.session_state.current_local_path = local_model_path
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
                return None
    
    return st.session_state.current_model

# Main content
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose an image of a handwritten document",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        help="Upload a scanned or photographed handwritten document"
    )

with col2:
    st.subheader("🎯 Selected Model")
    model_info = f"**Model:** {model_type}  \n"
    if local_model_path:
        import os
        model_info += f"**Mode:** Local Model  \n**Path:** {os.path.basename(local_model_path)}"
    else:
        model_info += f"**Mode:** {'Fine-tuned' if use_finetuned else 'Pre-trained'}"
    st.info(model_info)

if uploaded_file is not None:
    # Load and display original image
    st.markdown("---")
    st.subheader("📊 Image Processing")
    
    # Open image
    original_image = Image.open(uploaded_file)
    
    # Create two columns for original and processed
    orig_col, proc_col = st.columns(2)
    
    with orig_col:
        st.markdown("##### Original Image")
        st.image(original_image, use_column_width=True)
    
    # Preprocess image
    processed_image = preprocess_image(original_image)
    
    with proc_col:
        st.markdown("##### Processed Image")
        st.image(processed_image, use_column_width=True)
    
    # Preprocessing details
    with st.expander("📝 Preprocessing Details"):
        st.markdown("""
        The preprocessing pipeline applies the following steps:
        1. **Grayscale Conversion** - Reduces color information to improve text extraction
        2. **CLAHE Enhancement** - Contrast Limited Adaptive Histogram Equalization for better contrast
        3. **Bilateral Filtering** - Noise reduction while preserving text edges
        4. **Otsu Thresholding** - Creates a clean binary image for optimal OCR
        """)
    
    st.markdown("---")
    
    # OCR Processing
    st.subheader("🔤 Text Extraction")
    
    # Load model
    model = load_model_cached(model_type, use_finetuned, local_model_path)
    
    if model is not None:
        # Run OCR
        extract_button = st.button(
            "🚀 Extract Text",
            use_container_width=True,
            type="primary"
        )
        
        if extract_button:
            with st.spinner("Extracting text..."):
                try:
                    # Use fallback mechanism
                    extracted_text, inference_time, model_used = predict_with_fallback(
                        processed_image, 
                        model, 
                        use_gemini_fallback=use_gemini_fallback
                    )
                    
                    # Store in session state
                    st.session_state.extracted_text = extracted_text
                    st.session_state.inference_time = inference_time
                    st.session_state.model_used = model_used
                    
                except Exception as e:
                    st.error(f"Error during text extraction: {str(e)}")
        
        # Display results if available
        if "extracted_text" in st.session_state:
            # Check if Gemini fallback was used
            if "model_used" in st.session_state and "Fallback" in st.session_state.model_used:
                st.warning(f"⚠️ Primary model failed. Using {st.session_state.model_used}")
            else:
                st.success("✅ Text extraction completed!")
            
            # Display extracted text in copyable code block
            st.markdown("##### Extracted Text")
            st.code(st.session_state.extracted_text, language="text")
            
            # Copy button
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("📋 Copy to Clipboard", use_container_width=True):
                    st.write(st.session_state.extracted_text)
                    st.success("Copied!")
            
            with col2:
                # Download button
                st.download_button(
                    label="⬇️ Download Text",
                    data=st.session_state.extracted_text,
                    file_name="extracted_text.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Metrics
            st.subheader("📈 Performance Metrics")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    label="⚡ Inference Time",
                    value=f"{st.session_state.inference_time:.2f}",
                    delta="ms",
                    delta_color="off"
                )
            
            with metric_col2:
                char_count = len(st.session_state.extracted_text)
                st.metric(
                    label="📝 Characters Extracted",
                    value=f"{char_count}",
                    delta_color="off"
                )
            
            with metric_col3:
                word_count = len(st.session_state.extracted_text.split())
                st.metric(
                    label="💬 Words Extracted",
                    value=f"{word_count}",
                    delta_color="off"
                )
            
            # Efficiency note
            with st.expander("ℹ️ Efficiency Information"):
                throughput = f"{1000/st.session_state.inference_time:.1f}" if st.session_state.inference_time > 0 else "N/A"
                st.info(f"""
                **Inference Performance:**
                - **Time:** {st.session_state.inference_time:.2f}ms
                - **Device:** {'GPU (CUDA)' if 'cuda' in str(model.device) else 'CPU'}
                - **Throughput:** {throughput} images/second
                
                The model processes images efficiently using {'GPU acceleration' if 'cuda' in str(model.device) else 'CPU processing'}.
                """)

else:
    # Welcome message when no file is uploaded
    st.info("👆 Upload an image to get started!")
    
    st.markdown("""
    ### Features:
    - **Multiple OCR Models** - Choose between TrOCR and DocTR
    - **Image Preprocessing** - Automatic enhancement for better results
    - **Real-time Text Extraction** - Fast and accurate digitization
    - **Performance Metrics** - Monitor inference time and throughput
    - **Easy Export** - Copy or download extracted text
    
    ### Best Practices:
    - Ensure the document is well-lit and clearly visible
    - High contrast between text and background yields better results
    - Upload images with at least 300 DPI for optimal accuracy
    """)
