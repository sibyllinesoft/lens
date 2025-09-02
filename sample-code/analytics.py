# Data analytics module
import json
from typing import List, Dict, Any

class DataAnalyzer:
    def __init__(self):
        self.data = []
    
    def add_data_point(self, point: Dict[str, Any]) -> None:
        self.data.append(point)
    
    def analyze_trends(self, field: str) -> Dict[str, float]:
        values = [point.get(field) for point in self.data if field in point]
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        
        if not numeric_values:
            return {}
        
        return {
            'avg': sum(numeric_values) / len(numeric_values),
            'min': min(numeric_values),
            'max': max(numeric_values)
        }
    
    def search_data(self, query: str) -> List[Dict[str, Any]]:
        results = []
        query_lower = query.lower()
        
        for point in self.data:
            for key, value in point.items():
                if isinstance(value, str) and query_lower in value.lower():
                    results.append(point)
                    break
        
        return results