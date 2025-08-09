import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import re
import hashlib
from datetime import datetime
import streamlit as st

def format_bytes(bytes_value: int) -> str:
    """Convert bytes to human readable format"""
    try:
        if bytes_value == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = int(np.floor(np.log(bytes_value) / np.log(1024)))
        p = pow(1024, i)
        s = round(bytes_value / p, 2)
        
        return f"{s} {size_names[i]}"
    
    except (ValueError, OverflowError):
        return "Unknown size"

def validate_file_size(file_size: int, max_size_mb: int = 100) -> Tuple[bool, str]:
    """Validate uploaded file size"""
    try:
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if file_size > max_size_bytes:
            return False, f"File size ({format_bytes(file_size)}) exceeds maximum allowed size ({max_size_mb} MB)"
        
        return True, "File size is acceptable"
    
    except Exception as e:
        return False, f"Error validating file size: {str(e)}"

def sanitize_column_name(column_name: str) -> str:
    """Sanitize column names for safe processing"""
    try:
        # Convert to string
        sanitized = str(column_name)
        
        # Replace problematic characters
        sanitized = re.sub(r'[^\w\s-]', '_', sanitized)
        
        # Replace spaces with underscores
        sanitized = re.sub(r'\s+', '_', sanitized)
        
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = f"column_{hash(column_name) % 1000}"
        
        return sanitized
    
    except Exception:
        return f"column_{hash(str(column_name)) % 1000}"

def detect_data_types(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Detect and analyze data types in DataFrame"""
    try:
        type_analysis = {}
        
        for column in df.columns:
            series = df[column]
            
            analysis = {
                'pandas_dtype': str(series.dtype),
                'null_count': series.isnull().sum(),
                'null_percentage': (series.isnull().sum() / len(series)) * 100,
                'unique_count': series.nunique(),
                'sample_values': []
            }
            
            # Get sample values (non-null)
            non_null_values = series.dropna()
            if len(non_null_values) > 0:
                sample_size = min(5, len(non_null_values))
                analysis['sample_values'] = non_null_values.head(sample_size).tolist()
            
            # Type-specific analysis
            if series.dtype in ['int64', 'int32', 'float64', 'float32']:
                # Numeric analysis
                analysis.update({
                    'category': 'numeric',
                    'min_value': float(series.min()) if not series.empty else None,
                    'max_value': float(series.max()) if not series.empty else None,
                    'mean_value': float(series.mean()) if not series.empty else None,
                    'std_value': float(series.std()) if not series.empty else None,
                    'has_negatives': (series < 0).any(),
                    'has_zeros': (series == 0).any()
                })
                
                # Suggest better type if possible
                if series.dtype in ['float64', 'float32']:
                    if series.isnull().sum() == 0 and (series % 1 == 0).all():
                        analysis['suggested_type'] = 'integer'
                
            elif series.dtype == 'object':
                # Text analysis
                analysis.update({
                    'category': 'text',
                    'avg_length': series.astype(str).str.len().mean(),
                    'max_length': series.astype(str).str.len().max(),
                    'min_length': series.astype(str).str.len().min(),
                    'contains_numbers': series.astype(str).str.contains(r'\d').sum(),
                    'contains_special_chars': series.astype(str).str.contains(r'[^\w\s]').sum()
                })
                
                # Try to detect specific patterns
                analysis['patterns'] = detect_text_patterns(series)
                
                # Suggest better type if possible
                analysis['suggested_type'] = suggest_better_type(series)
                
            elif 'datetime' in str(series.dtype):
                # DateTime analysis
                analysis.update({
                    'category': 'datetime',
                    'min_date': str(series.min()) if not series.empty else None,
                    'max_date': str(series.max()) if not series.empty else None,
                    'date_range_days': (series.max() - series.min()).days if not series.empty else 0
                })
            
            else:
                analysis['category'] = 'other'
            
            type_analysis[column] = analysis
        
        return type_analysis
    
    except Exception as e:
        st.error(f"Error analyzing data types: {str(e)}")
        return {}

def detect_text_patterns(series: pd.Series) -> Dict[str, int]:
    """Detect common patterns in text data"""
    try:
        patterns = {
            'email_like': 0,
            'phone_like': 0,
            'url_like': 0,
            'date_like': 0,
            'numeric_string': 0,
            'all_caps': 0,
            'mixed_case': 0
        }
        
        # Sample for performance
        sample_size = min(1000, len(series))
        sample = series.dropna().head(sample_size)
        
        for value in sample:
            value_str = str(value)
            
            # Email pattern
            if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', value_str):
                patterns['email_like'] += 1
            
            # Phone pattern
            if re.search(r'(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', value_str):
                patterns['phone_like'] += 1
            
            # URL pattern
            if re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', value_str):
                patterns['url_like'] += 1
            
            # Date pattern
            if re.search(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', value_str):
                patterns['date_like'] += 1
            
            # Numeric string
            if value_str.replace('.', '').replace('-', '').isdigit():
                patterns['numeric_string'] += 1
            
            # Case patterns
            if value_str.isupper():
                patterns['all_caps'] += 1
            elif any(c.isupper() for c in value_str) and any(c.islower() for c in value_str):
                patterns['mixed_case'] += 1
        
        return patterns
    
    except Exception:
        return {}

def suggest_better_type(series: pd.Series) -> Optional[str]:
    """Suggest better data type for object columns"""
    try:
        # Sample for performance
        sample = series.dropna().head(100)
        
        if len(sample) == 0:
            return None
        
        # Try datetime conversion
        try:
            pd.to_datetime(sample, errors='raise')
            return 'datetime'
        except (ValueError, TypeError):
            pass
        
        # Try numeric conversion
        try:
            numeric_series = pd.to_numeric(sample, errors='raise')
            if (numeric_series % 1 == 0).all():
                return 'integer'
            else:
                return 'float'
        except (ValueError, TypeError):
            pass
        
        # Check if it's categorical (low cardinality)
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.5:
            return 'category'
        
        return None
    
    except Exception:
        return None

def calculate_data_profile_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive data profile summary"""
    try:
        summary = {
            'basic_info': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'duplicate_rows': df.duplicated().sum()
            },
            'column_types': {
                'numeric': len(df.select_dtypes(include=[np.number]).columns),
                'text': len(df.select_dtypes(include=['object']).columns),
                'datetime': len(df.select_dtypes(include=['datetime64']).columns),
                'boolean': len(df.select_dtypes(include=['bool']).columns),
                'other': len(df.select_dtypes(exclude=[np.number, 'object', 'datetime64', 'bool']).columns)
            },
            'data_quality': {
                'total_missing': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
                'complete_rows': len(df.dropna()),
                'complete_rows_percentage': (len(df.dropna()) / len(df)) * 100 if len(df) > 0 else 0
            },
            'uniqueness': {
                'columns_with_all_unique': sum(1 for col in df.columns if df[col].nunique() == len(df)),
                'columns_with_duplicates': sum(1 for col in df.columns if df[col].nunique() < len(df)),
                'avg_uniqueness_ratio': np.mean([df[col].nunique() / len(df) for col in df.columns]) if len(df) > 0 else 0
            }
        }
        
        # Calculate column-wise statistics
        column_stats = {}
        for col in df.columns:
            column_stats[col] = {
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'unique_percentage': (df[col].nunique() / len(df)) * 100 if len(df) > 0 else 0
            }
        
        summary['column_statistics'] = column_stats
        
        return summary
    
    except Exception as e:
        st.error(f"Error calculating data profile: {str(e)}")
        return {}

def validate_dataframe(df: pd.DataFrame, min_rows: int = 1, min_cols: int = 1) -> Tuple[bool, str]:
    """Validate DataFrame meets minimum requirements"""
    try:
        if df is None:
            return False, "DataFrame is None"
        
        if df.empty:
            return False, "DataFrame is empty"
        
        if len(df) < min_rows:
            return False, f"DataFrame has {len(df)} rows, minimum required: {min_rows}"
        
        if len(df.columns) < min_cols:
            return False, f"DataFrame has {len(df.columns)} columns, minimum required: {min_cols}"
        
        # Check for completely empty columns
        empty_columns = [col for col in df.columns if df[col].isnull().all()]
        if empty_columns:
            return False, f"DataFrame has completely empty columns: {empty_columns}"
        
        return True, "DataFrame validation passed"
    
    except Exception as e:
        return False, f"Error validating DataFrame: {str(e)}"

def clean_text_for_display(text: str, max_length: int = 100) -> str:
    """Clean and truncate text for safe display"""
    try:
        if not isinstance(text, str):
            text = str(text)
        
        # Remove or replace problematic characters
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        # Truncate if too long
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length - 3] + "..."
        
        return cleaned
    
    except Exception:
        return "Invalid text"

def generate_data_hash(df: pd.DataFrame) -> str:
    """Generate a hash for DataFrame content for caching purposes"""
    try:
        # Create a string representation of key DataFrame characteristics
        hash_string = f"{df.shape}_{df.columns.tolist()}_{df.dtypes.to_dict()}"
        
        # Add sample data for uniqueness
        if len(df) > 0:
            sample_data = df.head(5).to_string()
            hash_string += sample_data
        
        # Generate MD5 hash
        return hashlib.md5(hash_string.encode()).hexdigest()[:16]
    
    except Exception:
        return f"hash_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def format_number(value: Union[int, float], precision: int = 2) -> str:
    """Format numbers for display with appropriate precision"""
    try:
        if pd.isna(value):
            return "N/A"
        
        if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
            return f"{int(value):,}"
        
        # For floats
        if abs(value) >= 1000:
            return f"{value:,.{precision}f}"
        elif abs(value) >= 1:
            return f"{value:.{precision}f}"
        elif abs(value) >= 0.01:
            return f"{value:.{precision+1}f}"
        else:
            # Use scientific notation for very small numbers
            return f"{value:.{precision}e}"
    
    except Exception:
        return str(value)

def create_download_link(data: Union[str, bytes], filename: str, mime_type: str = 'text/plain') -> None:
    """Create a download link using Streamlit's download button"""
    try:
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        st.download_button(
            label=f"📥 Download {filename}",
            data=data_bytes,
            file_name=filename,
            mime=mime_type
        )
    
    except Exception as e:
        st.error(f"Error creating download link: {str(e)}")

def safe_eval_expression(expression: str, allowed_names: Dict[str, Any]) -> Any:
    """Safely evaluate simple expressions with restricted namespace"""
    try:
        # Define safe built-ins
        safe_builtins = {
            'abs': abs,
            'max': max,
            'min': min,
            'sum': sum,
            'len': len,
            'round': round,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool
        }
        
        # Combine with allowed names
        safe_dict = {**safe_builtins, **allowed_names}
        
        # Evaluate expression
        return eval(expression, {"__builtins__": {}}, safe_dict)
    
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{expression}': {str(e)}")

def get_memory_usage_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Get detailed memory usage information for DataFrame"""
    try:
        memory_info = {}
        
        # Overall memory usage
        total_memory = df.memory_usage(deep=True).sum()
        memory_info['total_mb'] = total_memory / 1024 / 1024
        memory_info['total_formatted'] = format_bytes(total_memory)
        
        # Per column memory usage
        column_memory = df.memory_usage(deep=True)
        memory_info['by_column'] = {}
        
        for col, memory in column_memory.items():
            memory_info['by_column'][col] = {
                'bytes': memory,
                'mb': memory / 1024 / 1024,
                'formatted': format_bytes(memory),
                'percentage': (memory / total_memory) * 100 if total_memory > 0 else 0
            }
        
        # Memory per row
        memory_info['per_row'] = total_memory / len(df) if len(df) > 0 else 0
        memory_info['per_row_formatted'] = format_bytes(memory_info['per_row'])
        
        return memory_info
    
    except Exception as e:
        st.error(f"Error calculating memory usage: {str(e)}")
        return {}

def estimate_processing_time(df: pd.DataFrame, operation_type: str = 'general') -> str:
    """Estimate processing time based on data size and operation type"""
    try:
        row_count = len(df)
        col_count = len(df.columns)
        
        # Base time estimates (in seconds) for different operations
        time_factors = {
            'general': 0.001,
            'cleaning': 0.002,
            'analysis': 0.005,
            'modeling': 0.01,
            'visualization': 0.001
        }
        
        factor = time_factors.get(operation_type, 0.001)
        estimated_seconds = (row_count * col_count * factor)
        
        if estimated_seconds < 1:
            return "< 1 second"
        elif estimated_seconds < 60:
            return f"{estimated_seconds:.0f} seconds"
        elif estimated_seconds < 3600:
            return f"{estimated_seconds/60:.1f} minutes"
        else:
            return f"{estimated_seconds/3600:.1f} hours"
    
    except Exception:
        return "Unknown"

def check_system_resources() -> Dict[str, Any]:
    """Check available system resources (simplified version)"""
    try:
        import psutil
        
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0,
            'available_memory_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
        }
    
    except ImportError:
        return {
            'cpu_percent': 0,
            'memory_percent': 0,
            'disk_percent': 0,
            'available_memory_gb': 0,
            'note': 'psutil not available - resource monitoring disabled'
        }
    
    except Exception as e:
        return {
            'error': f"Error checking system resources: {str(e)}"
        }
