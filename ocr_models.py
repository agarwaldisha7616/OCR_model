import torch
import time
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np
import os
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OCRModel:
    """Base class for OCR models."""
    
    def __init__(self):
        self.device = device
        self.model = None
        self.processor = None
    
    def predict(self, image: Image.Image) -> tuple[str, float]:
        """
        Predict text from image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (extracted_text, inference_time_ms)
        """
        raise NotImplementedError


class TrOCRModel(OCRModel):
    """TrOCR model implementation."""
    
    def __init__(self, model_name: str = "microsoft/trocr-large-handwritten", use_finetuned: bool = False, local_model_path: str = None):
        super().__init__()
        
        # Priority: local_model_path > use_finetuned > default model_name
        if local_model_path:
            try:
                print(f"Loading model from local path: {local_model_path}")
                self._load_from_local(local_model_path)
            except Exception as e:
                print(f"Failed to load model from {local_model_path}: {e}")
                print("Falling back to pretrained model...")
                self._load_pretrained(model_name)
        elif use_finetuned:
            try:
                self.processor = TrOCRProcessor.from_pretrained("trocr-finetuned")
                self.model = VisionEncoderDecoderModel.from_pretrained("trocr-finetuned").to(self.device)
            except Exception as e:
                print(f"Failed to load fine-tuned model: {e}. Using pretrained model instead.")
                self._load_pretrained(model_name)
        else:
            self._load_pretrained(model_name)
        
        self.model.eval()
    
    def _load_pretrained(self, model_name: str):
        """Load pretrained model from Hugging Face."""
        self.processor = TrOCRProcessor.from_pretrained(
            model_name,
            image_processor_kwargs={"do_rescale": False, "size": {"height": 384, "width": 384}}
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
    
    def _load_from_local(self, local_path: str):
        """Load model from local .safetensors file or directory."""
        import os
        
        # Check if path is a directory or file
        if os.path.isdir(local_path):
            # Load from directory containing model files
            self.processor = TrOCRProcessor.from_pretrained(
                local_path,
                image_processor_kwargs={"do_rescale": False, "size": {"height": 384, "width": 384}}
            )
            self.model = VisionEncoderDecoderModel.from_pretrained(
                local_path,
                local_files_only=True,
                device_map=self.device if torch.cuda.is_available() else "cpu"
            )
        else:
            # If it's a single .safetensors file, load the directory containing it
            model_dir = os.path.dirname(local_path)
            
            # Try to load processor from the same directory
            try:
                self.processor = TrOCRProcessor.from_pretrained(
                    model_dir,
                    image_processor_kwargs={"do_rescale": False, "size": {"height": 384, "width": 384}}
                )
            except Exception:
                # Fallback to default processor if not found
                print("Processor not found in local directory, using default processor...")
                self.processor = TrOCRProcessor.from_pretrained(
                    "microsoft/trocr-large-handwritten",
                    image_processor_kwargs={"do_rescale": False, "size": {"height": 384, "width": 384}}
                )
            
            # Load model from directory with proper device handling
            self.model = VisionEncoderDecoderModel.from_pretrained(
                model_dir,
                local_files_only=True,
                device_map=self.device if torch.cuda.is_available() else "cpu"
            )

    
    def predict(self, image: Image.Image) -> tuple[str, float]:
        """
        Predict text using TrOCR.
        
        Args:
            image: PIL Image object (should be in RGB mode)
            
        Returns:
            Tuple of (extracted_text, inference_time_ms)
        """
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Measure inference time
        start_time = time.time()
        
        with torch.no_grad():
            # Process image
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            
            # Generate text with better parameters for handwriting
            generated_ids = self.model.generate(
                pixel_values, 
                max_length=256,
                num_beams=5,
                early_stopping=True
            )
            
            # Decode text
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return text, inference_time


class DocTRModel(OCRModel):
    """DocTR model implementation (requires doctr package)."""
    
    def __init__(self):
        super().__init__()
        try:
            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor
            
            self.ocr_model = ocr_predictor(pretrained=True)
        except ImportError:
            raise ImportError("DocTR is not installed. Install with: pip install python-doctr")
    
    def predict(self, image: Image.Image) -> tuple[str, float]:
        """
        Predict text using DocTR.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (extracted_text, inference_time_ms)
        """
        import numpy as np
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Measure inference time
        start_time = time.time()
        
        # Run OCR
        doc_result = self.ocr_model([img_array])
        
        # Extract text from result
        text_lines = []
        for page in doc_result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        text_lines.append(word.value)
        
        text = " ".join(text_lines)
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return text, inference_time


class GeminiOCRModel(OCRModel):
    """Gemini model implementation for OCR fallback."""
    
    def __init__(self, api_key: str = None):
        super().__init__()
        try:
            import google.generativeai as genai
            
            # Get API key from environment variable if not provided
            if api_key is None:
                api_key = os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            
        except ImportError:
            raise ImportError("Google Generative AI is not installed. Install with: pip install google-generativeai")
    
    def predict(self, image: Image.Image) -> tuple[str, float]:
        """
        Predict text using Gemini Vision.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (extracted_text, inference_time_ms)
        """
        # Measure inference time
        start_time = time.time()
        
        try:
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Create prompt for OCR
            prompt = "Extract all text from this handwritten document image. Return only the text without any explanations or formatting."
            
            # Generate response
            response = self.model.generate_content([prompt, image])
            text = response.text.strip()
            
        except Exception as e:
            text = f"Gemini OCR failed: {str(e)}"
        
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return text, inference_time


def load_ocr_model(model_type: str = "TrOCR", use_finetuned: bool = False, local_model_path: str = None) -> OCRModel:
    """
    Load OCR model based on type.
    
    Args:
        model_type: Either "TrOCR", "DocTR", or "Gemini"
        use_finetuned: Whether to use fine-tuned version (only for TrOCR)
        local_model_path: Path to local model directory or .safetensors file (only for TrOCR)
        
    Returns:
        OCR model instance
    """
    if model_type == "TrOCR":
        return TrOCRModel(use_finetuned=use_finetuned, local_model_path=r"D:\projects\Handwriting Recognition\model\model.safetensors")
    elif model_type == "DocTR":
        return DocTRModel()
    elif model_type == "Gemini":
        return GeminiOCRModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict_with_fallback(image: Image.Image, primary_model: OCRModel, use_gemini_fallback: bool = True) -> tuple[str, float, str]:
    """
    Predict text with Gemini as fallback if primary model fails.
    
    Args:
        image: PIL Image object
        primary_model: Primary OCR model to use first
        use_gemini_fallback: Whether to use Gemini as fallback if primary fails
        
    Returns:
        Tuple of (extracted_text, inference_time_ms, model_used)
    """
    try:
        # Try primary model
        text, inference_time = primary_model.predict(image)
        
        # Check if result is valid (not empty and not too short)
        if text and len(text.strip()) > 0:
            return text, inference_time, type(primary_model).__name__
        else:
            raise ValueError("Primary model returned empty text")
            
    except Exception as e:
        print(f"Primary model failed: {str(e)}")
        
        if use_gemini_fallback:
            print("Falling back to Gemini...")
            try:
                gemini_model = GeminiOCRModel()
                text, inference_time = gemini_model.predict(image)
                return text, inference_time, "GeminiOCRModel (Fallback)"
            except Exception as gemini_error:
                return f"Both models failed. Primary: {str(e)}, Gemini: {str(gemini_error)}", 0.0, "Failed"
        else:
            return f"OCR failed: {str(e)}", 0.0, "Failed"

