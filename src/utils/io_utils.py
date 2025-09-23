from pathlib import Path
import json
import yaml
import os
from typing import Any, Dict, Optional, Union

def save_json(obj: Any, path: Union[str, Path]) -> None:
    """Save object to JSON file.
    
    Args:
        obj: Object to save
        path: Path to save the JSON file
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: Union[str, Path]) -> Any:
    """Load object from JSON file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Loaded object
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        path: Path to save the YAML file
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)

def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        path: Path to the YAML file
        
    Returns:
        Loaded configuration dictionary
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure that a directory exists, create if it doesn't.
    
    Args:
        path: Path to the directory
        
    Returns:
        Path object of the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
