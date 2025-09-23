"""AI-based paragraph detection using Tesseract OCR for text block detection."""
import cv2
import numpy as np
import pytesseract
import os

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\user\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

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

class AIParagraphDetector:
    """Paragraph detector using Tesseract OCR for text block detection."""
    
    def __init__(self, min_confidence: float = 0.7, min_block_size: int = 100):
        """
        Initialize the paragraph detector.
        
        Args:
            min_confidence: Minimum confidence score for detections (0-1)
            min_block_size: Minimum size of a text block in pixels
        """
        self.min_confidence = min_confidence
        self.min_block_size = min_block_size
        
        # Check if Tesseract is installed
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR initialized successfully")
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract is not installed or not in PATH")
            raise

    def detect_paragraphs(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect paragraphs in the document image."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Get OCR data with bounding boxes
        try:
            # Use Tesseract to detect text blocks
            d = pytesseract.image_to_data(
                gray,
                output_type=pytesseract.Output.DICT,
                config='--psm 6'  # Assume a single uniform block of text
            )
            
            blocks = []
            
            # Process each detected text block
            for i in range(len(d['level'])):
                if int(d['conf'][i]) > 0:  # Only consider confident detections
                    x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                    conf = float(d['conf'][i]) / 100.0  # Convert to 0-1 range
                    
                    # Skip small blocks
                    if w < self.min_block_size and h < self.min_block_size:
                        continue
                        
                    # Skip low confidence detections
                    if conf < self.min_confidence:
                        continue
                    
                    blocks.append(BoundingBox(
                        x=x, y=y, w=w, h=h,
                        confidence=conf,
                        text=d['text'][i].strip()
                    ))
            
            # Merge nearby blocks into paragraphs
            merged_blocks = self._merge_nearby_blocks(blocks)
            
            return merged_blocks
            
        except Exception as e:
            logger.error(f"Error in Tesseract OCR: {e}")
            return []

    def _merge_nearby_blocks(self, blocks: List[BoundingBox], 
                           x_threshold: float = 0.5, 
                           y_threshold: float = 1.5) -> List[BoundingBox]:
        """Merge text blocks that are close to each other into paragraphs."""
        if not blocks:
            return []
            
        # Sort blocks by y-coordinate, then by x-coordinate
        blocks.sort(key=lambda b: (b.y, b.x))
        
        merged = [blocks[0]]
        
        for block in blocks[1:]:
            last = merged[-1]
            
            # Calculate vertical and horizontal overlaps
            y_overlap = (min(block.y + block.h, last.y + last.h) - 
                        max(block.y, last.y)) / min(block.h, last.h)
            
            x_distance = max(0, block.x - (last.x + last.w))
            x_overlap = 1 - (x_distance / max(block.w, last.w))
            
            # Merge if blocks are vertically aligned and close enough
            if (y_overlap > 0.3 and x_overlap > x_threshold) or y_overlap > y_threshold:
                # Merge the blocks
                x = min(last.x, block.x)
                y = min(last.y, block.y)
                w = max(last.x + last.w, block.x + block.w) - x
                h = max(last.y + last.h, block.y + block.h) - y
                conf = (last.confidence + block.confidence) / 2
                text = f"{last.text} {block.text}".strip()
                
                merged[-1] = BoundingBox(x, y, w, h, conf, text)
            else:
                merged.append(block)
                
        return merged

def detect_paragraphs(
    image: np.ndarray,
    min_confidence: float = 0.7,
    min_block_size: int = 100
) -> List[BoundingBox]:
    """Convenience function to detect paragraphs in an image.
    
    Args:
        image: Input image (BGR or grayscale)
        min_confidence: Minimum confidence score (0-1)
        min_block_size: Minimum size of text blocks to consider
        
    Returns:
        List of detected paragraphs as BoundingBox objects
    """
    detector = AIParagraphDetector(
        min_confidence=min_confidence,
        min_block_size=min_block_size
    )
    return detector.detect_paragraphs(image)
