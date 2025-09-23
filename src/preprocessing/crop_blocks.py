import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict, Any, Literal
from tqdm import tqdm
import os
import imutils
import unicodedata
from dataclasses import dataclass

# Import AI-based detector (optional)
try:
    from .ai_paragraph_detector import AIParagraphDetector, BoundingBox as AIBoundingBox
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logger.warning("AI paragraph detector not available. Using traditional method.")
    class AIBoundingBox:
        pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    """Class to represent a bounding box with confidence score."""
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0
    text: str = ""
    
    @property
    def area(self) -> int:
        return self.w * self.h
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'x': self.x,
            'y': self.y,
            'width': self.w,
            'height': self.h,
            'confidence': self.confidence,
            'text': self.text
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoundingBox':
        return cls(
            x=data['x'],
            y=data['y'],
            w=data['width'],
            h=data['height'],
            confidence=data.get('confidence', 1.0),
            text=data.get('text', '')
        )

class DocumentPreprocessor:
    """Class for preprocessing document images and detecting text blocks."""
    
    def __init__(
        self,
        min_block_height: int = 20,
        min_block_width: int = 100,
        max_skew_angle: float = 5.0,
        debug: bool = False,
        use_ai_detector: bool = True,
        ai_confidence: float = 0.7
    ):
        """Initialize the document preprocessor.
        
        Args:
            min_block_height: Minimum height of a text block in pixels (used for non-AI detection)
            min_block_width: Minimum width of a text block in pixels (used for non-AI detection)
            max_skew_angle: Maximum skew angle to detect and correct (in degrees)
            debug: Whether to save intermediate processing results for debugging
            use_ai_detector: Whether to use AI-based paragraph detection if available
            ai_confidence: Confidence threshold for AI detections (0.0 to 1.0)
        """
        self.min_block_height = min_block_height
        self.min_block_width = min_block_width
        self.max_skew_angle = max_skew_angle
        self.debug = debug
        self.use_ai_detector = use_ai_detector and AI_AVAILABLE
        self.ai_confidence = ai_confidence
        
        if self.use_ai_detector:
            try:
                self.ai_detector = AIParagraphDetector(min_confidence=ai_confidence)
                logger.info("AI paragraph detector initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize AI detector: {e}")
                self.use_ai_detector = False
    
    def preprocess_image(
        self, 
        image: np.ndarray, 
        denoise: bool = True, 
        adaptive_thresh: bool = True,
        dilate: bool = True
    ) -> np.ndarray:
        """Preprocess the image for better text detection."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply denoising
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Apply adaptive thresholding
        if adaptive_thresh:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if needed (text should be white, background black)
        if np.mean(binary) > 128:
            binary = cv2.bitwise_not(binary)
        
        # Apply dilation to connect text components
        if dilate:
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=1)
        
        return binary
    
    def detect_skew(self, image: np.ndarray) -> float:
        """Detect the skew angle of the document."""
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Find coordinates of non-zero pixels
        coords = np.column_stack(np.where(thresh > 0))
        
        # Calculate minimum area rectangle and angle
        angle = cv2.minAreaRect(coords)[-1]
        
        # Handle angle range
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        return angle
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Deskew the image based on detected skew angle."""
        if len(image.shape) == 3:
            h, w = image.shape[:2]
        else:
            h, w = image.shape
            
        # Calculate center of the image
        center = (w // 2, h // 2)
        
        # Calculate the rotation matrix
        M = cv2.getRotationMatrix2D(center, self.max_skew_angle, 1.0)
        
        # Perform the rotation
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def detect_text_blocks(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect paragraph blocks in the document image.
        
        Uses AI-based detection if available and enabled, otherwise falls back to
        traditional computer vision methods.
        """
        if self.use_ai_detector:
            return self._detect_paragraphs_ai(image)
        else:
            return self._detect_paragraphs_cv(image)
    
    def _detect_paragraphs_ai(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect paragraphs using AI-based detection."""
        try:
            # Convert to BGR if needed (OpenCV default)
            if len(image.shape) == 2:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image_bgr = image.copy()
                
            # Get AI detections
            ai_boxes = self.ai_detector.detect_paragraphs(image_bgr)
            
            # Convert AI BoundingBox to our format
            blocks = []
            for box in ai_boxes:
                blocks.append(BoundingBox(
                    x=box.x, 
                    y=box.y, 
                    w=box.w, 
                    h=box.h, 
                    confidence=box.confidence
                ))
                
            return blocks
            
        except Exception as e:
            logger.error(f"AI detection failed: {e}")
            logger.warning("Falling back to traditional detection method")
            self.use_ai_detector = False
            return self._detect_paragraphs_cv(image)
    
    def _detect_paragraphs_cv(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect paragraphs using traditional computer vision methods."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate image dimensions for relative sizing
        img_h, img_w = gray.shape[:2]
        min_paragraph_height = max(50, int(img_h * 0.03))  # At least 3% of image height
        min_paragraph_width = max(100, int(img_w * 0.2))   # At least 20% of image width
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 10
        )
        
        # Use morphological operations to connect lines into paragraphs
        # Horizontal kernel to connect words in the same line
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        # Vertical kernel to connect lines into paragraphs
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        
        # Connect text horizontally (words into lines)
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
        # Connect lines vertically (lines into paragraphs)
        vertical = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, vertical_kernel, iterations=3)
        
        # Find contours of paragraph blocks
        contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blocks = []
        
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip small blocks (likely single lines or noise)
            if w < min_paragraph_width or h < min_paragraph_height:
                continue
                
            # Skip blocks that are too narrow (likely vertical lines or noise)
            if w < h * 0.5:  # Width less than half the height
                continue
                
            # Calculate confidence based on size and aspect ratio
            area_ratio = (w * h) / (img_w * img_h)
            aspect_ratio = w / float(h) if h != 0 else 0
            
            # Higher confidence for blocks with reasonable aspect ratio and size
            confidence = min(1.0, area_ratio * 100)
            
            # Add padding (percentage of the block size)
            h_pad = int(h * 0.1)  # 10% padding
            w_pad = int(w * 0.1)  # 10% padding
            
            x = max(0, x - w_pad)
            y = max(0, y - h_pad)
            w = min(img_w - x, w + 2 * w_pad)
            h = min(img_h - y, h + 2 * h_pad)
            
            blocks.append(BoundingBox(x, y, w, h, confidence))
        
        # Sort blocks from top to bottom, left to right
        blocks.sort(key=lambda b: (b.y // (min_paragraph_height * 2), b.x))
        
        return blocks
    
    def visualize_blocks(self, image: np.ndarray, blocks: List[BoundingBox], output_path: Optional[str] = None) -> np.ndarray:
        """Visualize detected text blocks on the image."""
        vis = image.copy()
        
        for i, block in enumerate(blocks):
            # Draw rectangle
            color = (0, 255, 0)  # Green
            cv2.rectangle(vis, (block.x, block.y), (block.x + block.w, block.y + block.h), color, 2)
            
            # Add block number
            cv2.putText(vis, str(i + 1), (block.x, block.y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, vis)
            
        return vis

def normalize_filename(filename: str) -> str:
    """Normalize filename to handle non-ASCII characters."""
    # Normalize unicode characters
    normalized = unicodedata.normalize('NFKD', str(filename))
    # Remove any characters that can't be encoded in the filesystem
    safe_name = normalized.encode('ascii', 'ignore').decode('ascii')
    # Replace any remaining problematic characters with underscore
    safe_name = "_".join(safe_name.split())
    return safe_name or "unnamed"

def crop_blocks(
    page_files: Union[str, Path, List[Union[str, Path]]],
    out_dir: Union[str, Path] = "outputs/crops",
    min_block_height: int = 20,
    min_block_width: int = 100,
    max_skew_angle: float = 5.0,
    debug: bool = False,
    progress_bar: bool = True,
    use_ai_detector: bool = True,
    ai_confidence: float = 0.7
) -> List[Path]:
    """
    Split page images into text blocks (paragraphs) using advanced document analysis.
    
    Args:
        page_files: Path to a single image file or directory of images, or list of image paths
        out_dir: Directory to save the cropped blocks
        min_block_height: Minimum height of a text block in pixels
        min_block_width: Minimum width of a text block in pixels
        max_skew_angle: Maximum skew angle to detect and correct (in degrees)
        debug: Whether to save intermediate processing results for debugging
        progress_bar: Whether to show a progress bar
        use_ai_detector: Whether to use AI-based paragraph detection
        ai_confidence: Minimum confidence score for AI-based paragraph detection
        
    Returns:
        List of paths to the saved cropped blocks
    """
    # Convert to list if single path is provided
    if isinstance(page_files, (str, Path)):
        page_files = [Path(page_files)]
    else:
        page_files = [Path(f) for f in page_files]
    
    # Create output directory
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize document preprocessor
    preprocessor = DocumentPreprocessor(
        min_block_height=min_block_height,
        min_block_width=min_block_width,
        max_skew_angle=max_skew_angle,
        debug=debug,
        use_ai_detector=use_ai_detector,
        ai_confidence=ai_confidence
    )
    
    # Create debug directory if needed
    if debug:
        debug_dir = Path(out_dir) / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
    
    crop_paths = []
    
    # Process each page
    for page_path in tqdm(page_files, desc="Processing pages", disable=not progress_bar):
        try:
            # Normalize the page path to handle non-ASCII characters
            normalized_stem = normalize_filename(page_path.stem)
            
            # Read the image
            image = cv2.imread(str(page_path))
            if image is None:
                logger.warning(f"Could not read image: {page_path}")
                continue
                
            # Create output directory for this page with normalized name
            page_dir = out_dir / normalized_stem
            page_dir.mkdir(parents=True, exist_ok=True)
            
            # Detect and crop text blocks
            blocks = preprocessor.detect_text_blocks(image)
            
            # Save debug visualization if enabled
            if debug:
                debug_page_dir = debug_dir / normalized_stem
                debug_page_dir.mkdir(parents=True, exist_ok=True)
                debug_path = debug_page_dir / f"{normalized_stem}_blocks.jpg"
                
                # Create visualization with block numbers and confidence scores
                vis = image.copy()
                for i, block in enumerate(blocks):
                    # Draw rectangle
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(vis, (block.x, block.y), 
                                (block.x + block.w, block.y + block.h), 
                                color, 2)
                    
                    # Add block number and confidence
                    label = f"{i+1} ({block.confidence:.2f})"
                    cv2.putText(vis, label, 
                              (block.x, block.y - 5 if block.y > 20 else block.y + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                cv2.imwrite(str(debug_path), vis)
            
            # Save each block
            for i, block in enumerate(blocks):
                try:
                    # Extract the block
                    block_img = image[block.y:block.y+block.h, block.x:block.x+block.w]
                    
                    # Generate output filename with normalized name
                    block_filename = f"{normalized_stem}_block_{i+1:03d}.png"
                    block_path = page_dir / block_filename
                    
                    # Save the block
                    success = cv2.imwrite(str(block_path), block_img)
                    if not success:
                        logger.error(f"Failed to save block {i+1} in {normalized_stem}")
                        continue
                        
                    crop_paths.append(block_path)
                    
                except Exception as e:
                    logger.error(f"Error processing block {i+1} in {normalized_stem}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing {page_path.name}: {str(e)}")
            continue
    
    return crop_paths
