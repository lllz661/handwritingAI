import editdistance
from typing import Dict, List, Tuple, Union, Optional
import numpy as np

def cer(pred: str, gt: str) -> float:
    """Calculate Character Error Rate (CER).
    
    Args:
        pred: Predicted text
        gt: Ground truth text
        
    Returns:
        CER as a float between 0 and 1
    """
    if not gt:
        return 0.0 if not pred else 1.0
    if not pred:
        return 1.0
    return editdistance.eval(pred, gt) / max(1, len(gt))

def wer(pred: str, gt: str) -> float:
    """Calculate Word Error Rate (WER).
    
    Args:
        pred: Predicted text
        gt: Ground truth text
        
    Returns:
        WER as a float between 0 and 1
    """
    g = gt.split()
    p = pred.split()
    if not g:
        return 0.0 if not p else 1.0
    if not p:
        return 1.0
    return editdistance.eval(p, g) / max(1, len(g))

def accuracy(pred: str, gt: str) -> float:
    """Calculate word-level accuracy.
    
    Args:
        pred: Predicted text
        gt: Ground truth text
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    return 1.0 if pred.strip() == gt.strip() else 0.0

def calculate_metrics(
    predictions: List[str], 
    ground_truths: List[str],
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """Calculate various text similarity metrics.
    
    Args:
        predictions: List of predicted texts
        ground_truths: List of ground truth texts
        metrics: List of metrics to calculate. Defaults to ['cer', 'wer', 'accuracy']
        
    Returns:
        Dictionary with metric names as keys and average scores as values
    """
    if metrics is None:
        metrics = ['cer', 'wer', 'accuracy']
        
    if len(predictions) != len(ground_truths):
        raise ValueError("Length of predictions and ground_truths must be the same")
    
    results = {metric: [] for metric in metrics}
    
    for pred, gt in zip(predictions, ground_truths):
        if 'cer' in metrics:
            results['cer'].append(cer(pred, gt))
        if 'wer' in metrics:
            results['wer'].append(wer(pred, gt))
        if 'accuracy' in metrics:
            results['accuracy'].append(accuracy(pred, gt))
    
    # Calculate average for each metric
    avg_results = {}
    for metric, values in results.items():
        if values:  # Only include metrics that were calculated
            avg_results[metric] = float(np.mean(values))
    
    return avg_results
