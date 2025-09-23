import fitz
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
import cv2
import numpy as np
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_pdf(pdf_path: Union[str, Path]) -> Tuple[bool, str]:
    """Validate if the input file is a valid PDF."""
    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return False, f"File not found: {pdf_path}"
        if pdf_path.suffix.lower() != '.pdf':
            return False, f"File is not a PDF: {pdf_path}"
        with fitz.open(str(pdf_path)) as _:
            return True, ""
    except Exception as e:
        return False, f"Invalid PDF file: {str(e)}"

def preprocess_image(image: np.ndarray, denoise: bool = True, threshold: bool = True) -> np.ndarray:
    """Preprocess the image for better OCR results."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if denoise:
        image = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    if threshold:
        image = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    
    return image

def pdf_to_images(
    pdf_path: Union[str, Path],
    out_dir: Union[str, Path] = "outputs/pages",
    dpi: int = 200,
    preprocess: bool = True,
    progress_bar: bool = True
) -> List[Path]:
    """
    Convert each page of a PDF into PNG images with optional preprocessing.
    
    Args:
        pdf_path: Path to the input PDF file
        out_dir: Directory to save the output images
        dpi: DPI for the output images (higher = better quality but larger files)
        preprocess: Whether to apply image preprocessing
        progress_bar: Whether to show a progress bar
        
    Returns:
        List of paths to the saved page images
        
    Raises:
        FileNotFoundError: If the input PDF doesn't exist
        ValueError: If the input file is not a valid PDF
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)

    is_valid, error_msg = validate_pdf(pdf_path)
    if not is_valid:
        logger.error(f"PDF validation failed: {error_msg}")
        raise ValueError(f"Invalid PDF file: {error_msg}")

    out_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    try:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        
        if total_pages == 0:
            logger.warning("PDF contains no pages")
            return []
            
        logger.info(f"Processing {total_pages} pages from {pdf_path.name}")

        if progress_bar:
            pbar = tqdm(total=total_pages, desc="Converting pages", unit="page")

        for i in range(total_pages):
            try:
                page = doc[i]
                zoom = dpi / 72  # 72 DPI is the default in PyMuPDF
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                if preprocess:

                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, pix.n
                    )
                    
                    # Apply preprocessing
                    img = preprocess_image(img)
                    
                    # Save the preprocessed image
                    out_file = out_dir / f"{pdf_path.stem}_page_{i+1:03d}.png"
                    cv2.imwrite(str(out_file), img)
                else:
                    # Save the original image
                    out_file = out_dir / f"{pdf_path.stem}_page_{i+1:03d}.png"
                    pix.save(str(out_file))
                
                saved_paths.append(out_file)
                
                # Update progress
                if progress_bar:
                    pbar.update(1)
                
            except Exception as e:
                logger.error(f"Error processing page {i+1}: {str(e)}")
                continue
        
        if progress_bar:
            pbar.close()
            
        logger.info(f"Successfully converted {len(saved_paths)}/{total_pages} pages to images"
                   f" in {out_dir.absolute()}")
        
        return saved_paths
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path.name}: {str(e)}")
        # Clean up partially converted files if there was an error
        for img_path in saved_paths:
            try:
                if img_path.exists():
                    os.remove(img_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up {img_path}: {str(cleanup_error)}")
        raise
    
    finally:
        # Ensure resources are properly closed
        if 'doc' in locals():
            doc.close()
        if progress_bar and 'pbar' in locals():
            pbar.close()
