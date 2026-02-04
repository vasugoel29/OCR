"""Token distribution analysis logic."""

from typing import Dict, Tuple
import re

class DistributionAnalyzer:
    """Analyzes token distribution statistics."""
    
    def __init__(self, config: Dict):
        """Initialize analyzer.
        
        Args:
            config: Configuration containing 'distribution' section
        """
        self.config = config.get('distribution', {})
        
    def analyze(self, text: str, document_type: str) -> Tuple[float, Dict]:
        """Analyze text distribution against profile.
        
        Args:
            text: Full text content
            document_type: Type of document
            
        Returns:
            Tuple of (DistributionScore, metrics_dict)
        """
        if not text:
            return 0.0, {'error': 'No text'}
            
        if document_type not in self.config:
            return 1.0, {}  # Pass if no profile defined
            
        profile = self.config[document_type]
        min_numeric = profile.get('min_numeric_ratio', 0.0)
        max_special = profile.get('max_special_char_ratio', 1.0)
        
        # Calculate metrics
        total_chars = len(text)
        if total_chars == 0:
            return 0.0, {}
            
        numeric_count = sum(c.isdigit() for c in text)
        alphanumeric_count = sum(c.isalnum() or c.isspace() for c in text)
        special_char_count = total_chars - alphanumeric_count
        
        numeric_ratio = numeric_count / total_chars
        special_char_ratio = special_char_count / total_chars
        
        # Calculate score (1.0 = matches profile, 0.0 = completely off)
        score = 1.0
        
        # Penalize low numeric ratio (for invoices)
        if numeric_ratio < min_numeric:
            deviation = (min_numeric - numeric_ratio) / min_numeric
            score -= deviation * 0.5  # Max 0.5 penalty
            
        # Penalize high special char ratio (garbage)
        if special_char_ratio > max_special:
            deviation = (special_char_ratio - max_special) / (1.0 - max_special)
            score -= deviation * 0.8  # Heavy penalty for garbage
            
        metrics = {
            'numeric_ratio': numeric_ratio,
            'special_char_ratio': special_char_ratio,
            'total_chars': total_chars
        }
        
        return max(0.0, score), metrics
