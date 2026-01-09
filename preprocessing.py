import cv2
import numpy as np
from PIL import Image


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Apply preprocessing to enhance handwritten text recognition.
    
    Steps:
    1. Convert to grayscale
    2. Apply CLAHE for contrast enhancement
    3. Apply bilateral filter for noise reduction
    4. Convert back to RGB for TrOCR (preserves details better than binary)
    
    Args:
        image: PIL Image object
        
    Returns:
        Processed PIL Image object in RGB
    """
    # Convert PIL to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter for noise reduction while preserving edges
    filtered = cv2.bilateralFilter(enhanced, 5, 50, 50)
    
    # Convert grayscale back to RGB (TrOCR expects RGB images)
    rgb_image = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
    
    # Convert back to PIL format
    processed_image = Image.fromarray(rgb_image)
    
    return processed_image


def preprocess_image_cv2(image: Image.Image) -> np.ndarray:
    """
    Apply preprocessing and return as OpenCV format (numpy array).
    
    Args:
        image: PIL Image object
        
    Returns:
        Processed image as numpy array
    """
    processed = preprocess_image(image)
    return cv2.cvtColor(np.array(processed), cv2.COLOR_GRAY2BGR)
