import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from rapidfuzz import fuzz
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TextBlock:
    """Class to represent a text block with its metadata."""
    image_path: str
    text: str
    page: str
    block_index: int
    confidence: float = 1.0
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        if self.metadata is None:
            del result['metadata']
        return result

class TextAligner:
    """Class for aligning cropped text images with ground truth text."""
    
    def __init__(
        self,
        min_text_length: int = 5,
        max_text_ratio: float = 3.0,
        min_similarity: float = 0.6,
        debug: bool = False
    ):
        """
        Initialize the text aligner.
        
        Args:
            min_text_length: Minimum length of text to consider for alignment
            max_text_ratio: Maximum allowed ratio of text lengths for matching
            min_similarity: Minimum similarity score (0-1) to consider a match
            debug: Whether to enable debug logging
        """
        self.min_text_length = min_text_length
        self.max_text_ratio = max_text_ratio
        self.min_similarity = min_similarity
        self.debug = debug
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for comparison."""
        # Convert to lowercase and remove extra whitespace
        text = ' '.join(str(text).lower().split())
        # Remove common punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
            
        # Clean the text
        clean1 = self.clean_text(text1)
        clean2 = self.clean_text(text2)
        
        # If either text is too short, use exact matching
        if len(clean1) < self.min_text_length or len(clean2) < self.min_text_length:
            return 1.0 if clean1 == clean2 else 0.0
        
        # Calculate length ratio
        len_ratio = max(len(clean1), len(clean2)) / (min(len(clean1), len(clean2)) + 1e-9)
        if len_ratio > self.max_text_ratio:
            return 0.0
        
        # Calculate text similarity using Levenshtein distance
        similarity = fuzz.ratio(clean1, clean2) / 100.0
        return similarity
    
    def read_ground_truth(self, gt_path: Path) -> List[Tuple[str, float]]:
        """
        Read ground truth text from a file with support for multiple formats.
        
        Returns:
            List of (text, confidence) tuples
        """
        try:
            content = gt_path.read_text(encoding='utf-8').strip()
            if not content:
                logger.warning(f"Empty ground truth file: {gt_path}")
                return []
                
            # Try different formats
            if gt_path.suffix.lower() == '.json':
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        return [(str(item.get('text', '')), float(item.get('confidence', 1.0))) 
                               for item in data if item.get('text', '').strip()]
                    elif isinstance(data, dict):
                        return [(str(text), 1.0) for text in data.values() if text.strip()]
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in ground truth file: {gt_path}")
            
            # Default: treat as plain text with paragraphs separated by double newlines
            paragraphs = []
            for para in content.split('\n\n'):
                para = para.strip()
                if para:
                    paragraphs.append((para, 1.0))
            
            # If no paragraphs found, try splitting by single newlines
            if not paragraphs and '\n' in content:
                paragraphs = [(line, 1.0) for line in content.splitlines() if line.strip()]
            
            return paragraphs
            
        except Exception as e:
            logger.error(f"Error reading ground truth file {gt_path}: {str(e)}")
            return []
    
    def align_blocks_to_text(
        self,
        image_paths: List[Union[str, Path]],
        gt_dir: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None
    ) -> Dict[str, TextBlock]:
        """
        Align cropped text images with ground truth text.
        
        Args:
            image_paths: List of paths to cropped text images
            gt_dir: Directory containing ground truth text files
            output_file: Optional path to save the alignment results
            
        Returns:
            Dictionary mapping image paths to TextBlock objects
        """
        gt_dir = Path(gt_dir)
        if not gt_dir.exists() or not gt_dir.is_dir():
            raise ValueError(f"Ground truth directory not found: {gt_dir}")
        
        # Group images by page
        pages: Dict[str, List[Path]] = {}
        for img_path in map(Path, image_paths):
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue
                
            # Extract page ID from filename (format: {page_id}_block_{index}.png)
            # Handle filenames with spaces and special characters
            match = re.search(r'^(.+?)(?:_page_\d+)?_block_\d+$', img_path.stem)
            if not match:
                logger.warning(f"Could not extract page ID from filename: {img_path}")
                continue
                
            page_id = match.group(1).strip()
            pages.setdefault(page_id, []).append(img_path)
        
        results = {}
        
        # Process each page
        for page_id, img_paths in tqdm(pages.items(), desc="Aligning pages"):
            # Find matching ground truth file
            gt_files = list(gt_dir.glob(f"{page_id}*"))
            if not gt_files:
                logger.warning(f"No ground truth found for page: {page_id}")
                continue
                
            gt_file = gt_files[0]
            gt_texts = self.read_ground_truth(gt_file)
            
            if not gt_texts:
                logger.warning(f"No valid ground truth text found in: {gt_file}")
                continue
            
            # Sort images by block index
            def get_block_index(p):
                match = re.search(r'_block_(\d+)(?:\.\w+)?$', p.name)
                return int(match.group(1)) if match else 0
                
            img_paths_sorted = sorted(img_paths, key=get_block_index)
            
            # Simple 1:1 mapping if counts match
            if len(img_paths_sorted) == len(gt_texts):
                for img_path, (text, conf) in zip(img_paths_sorted, gt_texts):
                    results[str(img_path)] = TextBlock(
                        image_path=str(img_path),
                        text=text,
                        page=page_id,
                        block_index=len([k for k in results if k.startswith(page_id)]),
                        confidence=conf
                    )
                continue
                
            # Otherwise, try to find best matches
            logger.info(f"Mismatched counts for {page_id}: {len(img_paths_sorted)} images, {len(gt_texts)} text blocks")
            
            # For simplicity, just take first N or all available
            count = min(len(img_paths_sorted), len(gt_texts))
            for i in range(count):
                results[str(img_paths_sorted[i])] = TextBlock(
                    image_path=str(img_paths_sorted[i]),
                    text=gt_texts[i][0],
                    page=page_id,
                    block_index=i,
                    confidence=gt_texts[i][1]
                )
        
        # Save results if output file is specified
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {k: v.to_dict() for k, v in results.items()},
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            logger.info(f"Saved alignment results to {output_file}")
        
        return results

def align_text(
    crop_files: List[Union[str, Path]],
    gt_dir: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    min_similarity: float = 0.6,
    debug: bool = False
) -> Dict[str, dict]:
    """
    Align cropped text images with ground truth text.
    
    Args:
        crop_files: List of paths to cropped text images
        gt_dir: Directory containing ground truth text files
        output_file: Path to save the alignment results (JSON)
        min_similarity: Minimum similarity score (0-1) to consider a match
        debug: Whether to enable debug logging
        
    Returns:
        Dictionary mapping image paths to text blocks
    """
    aligner = TextAligner(min_similarity=min_similarity, debug=debug)
    results = aligner.align_blocks_to_text(crop_files, gt_dir, output_file)
    return {k: v.to_dict() for k, v in results.items()}
