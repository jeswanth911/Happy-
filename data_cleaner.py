import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import hashlib

class DataCleaner:
    """Enterprise-grade data cleaning and standardization"""
    
    def __init__(self):
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
        
        self.column_name_mappings = {
            # Common variations to standard names
            'id': ['id', 'ID', 'Id', 'identifier', 'uid', 'key'],
            'name': ['name', 'Name', 'NAME', 'full_name', 'fullname', 'customer_name'],
            'email': ['email', 'Email', 'EMAIL', 'e_mail', 'mail', 'email_address'],
            'phone': ['phone', 'Phone', 'PHONE', 'telephone', 'mobile', 'phone_number'],
            'address': ['address', 'Address', 'ADDRESS', 'street', 'location'],
            'date': ['date', 'Date', 'DATE', 'timestamp', 'created_at', 'updated_at'],
            'price': ['price', 'Price', 'PRICE', 'cost', 'amount', 'value'],
            'quantity': ['quantity', 'Quantity', 'QUANTITY', 'qty', 'count', 'amount']
        }
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows with enterprise-level logic"""
        try:
            initial_count = len(df)
            
            # First, remove exact duplicates
            df_cleaned = df.drop_duplicates()
            
            # Then, handle near-duplicates for string columns
            string_columns = df.select_dtypes(include=['object']).columns
            
            if len(string_columns) > 0:
                # Create a similarity key for fuzzy matching
                df_cleaned = self._remove_fuzzy_duplicates(df_cleaned, list(string_columns))
            
            duplicates_removed = initial_count - len(df_cleaned)
            
            if duplicates_removed > 0:
                st.info(f"Removed {duplicates_removed} duplicate rows ({duplicates_removed/initial_count:.1%})")
            
            return df_cleaned
            
        except Exception as e:
            st.error(f"Error removing duplicates: {str(e)}")
            return df
    
    def _remove_fuzzy_duplicates(self, df: pd.DataFrame, string_columns: List[str]) -> pd.DataFrame:
        """Remove fuzzy duplicates based on string similarity"""
        try:
            # Create normalized versions of string columns for comparison
            df_temp = df.copy()
            
            for col in string_columns:
                if col in df_temp.columns:
                    # Normalize: lowercase, remove extra spaces, remove special chars
                    df_temp[f'{col}_normalized'] = (
                        df_temp[col].astype(str)
                        .str.lower()
                        .str.strip()
                        .str.replace(r'[^\w\s]', '', regex=True)
                        .str.replace(r'\s+', ' ', regex=True)
                    )
            
            # Find duplicates based on normalized columns
            normalized_cols = [f'{col}_normalized' for col in string_columns if f'{col}_normalized' in df_temp.columns]
            
            if normalized_cols:
                duplicates_mask = df_temp.duplicated(subset=normalized_cols, keep='first')
                df_cleaned = df[~duplicates_mask].copy()
                return df_cleaned
            
            return df
            
        except Exception as e:
            st.warning(f"Could not perform fuzzy duplicate removal: {str(e)}")
            return df
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names using enterprise conventions"""
        try:
            df_cleaned = df.copy()
            
            # Create mapping of old to new column names
            column_mapping = {}
            
            for col in df_cleaned.columns:
                # Standardize column name
                new_name = self._standardize_column_name(col)
                column_mapping[col] = new_name
            
            # Rename columns
            df_cleaned = df_cleaned.rename(columns=column_mapping)
            
            # Handle duplicate column names after standardization
            df_cleaned = self._handle_duplicate_columns(df_cleaned)
            
            return df_cleaned
            
        except Exception as e:
            st.error(f"Error standardizing columns: {str(e)}")
            return df
    
    def _standardize_column_name(self, col_name: str) -> str:
        """Standardize a single column name"""
        # Convert to lowercase
        name = str(col_name).lower()
        
        # Remove special characters and replace with underscores
        name = re.sub(r'[^\w\s]', '_', name)
        
        # Replace spaces with underscores
        name = re.sub(r'\s+', '_', name)
        
        # Remove multiple consecutive underscores
        name = re.sub(r'_+', '_', name)
        
        # Remove leading/trailing underscores
        name = name.strip('_')
        
        # Check for common mappings
        for standard_name, variations in self.column_name_mappings.items():
            if name in [v.lower() for v in variations]:
                return standard_name
        
        return name
    
    def _handle_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate column names by adding suffixes"""
        columns = list(df.columns)
        seen = {}
        new_columns = []
        
        for col in columns:
            if col in seen:
                seen[col] += 1
                new_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_columns.append(col)
        
        df.columns = new_columns
        return df
    
    def fix_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intelligently fix missing values based on data types and patterns"""
        try:
            df_cleaned = df.copy()
            
            for col in df_cleaned.columns:
                if df_cleaned[col].isnull().any():
                    df_cleaned[col] = self._fix_column_missing_values(df_cleaned[col])
            
            missing_summary = df.isnull().sum().sum() - df_cleaned.isnull().sum().sum()
            if missing_summary > 0:
                st.info(f"Fixed {missing_summary} missing values")
            
            return df_cleaned
            
        except Exception as e:
            st.error(f"Error fixing missing values: {str(e)}")
            return df
    
    def _fix_column_missing_values(self, series: pd.Series) -> pd.Series:
        """Fix missing values in a single column"""
        if series.isnull().all():
            return series
        
        # Determine data type and appropriate strategy
        if series.dtype in ['int64', 'float64']:
            # Numeric columns: use median for robust handling
            return series.fillna(series.median())
        
        elif series.dtype == 'object':
            # String columns: use mode (most frequent value)
            mode_value = series.mode()
            if len(mode_value) > 0:
                return series.fillna(mode_value[0])
            else:
                return series.fillna('Unknown')
        
        elif series.dtype in ['datetime64[ns]', 'datetime64[ns, UTC]']:
            # DateTime columns: forward fill then backward fill
            return series.ffill().bfill()
        
        else:
            # Default: forward fill
            return series.ffill()
    
    def detect_and_correct_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and correct anomalies using statistical methods"""
        try:
            df_cleaned = df.copy()
            anomalies_found = 0
            
            # Only process numeric columns for anomaly detection
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if len(df_cleaned[col].dropna()) > 10:  # Need sufficient data
                    original_anomalies = len(df_cleaned)
                    df_cleaned = self._detect_outliers_iqr(df_cleaned, col)
                    anomalies_found += original_anomalies - len(df_cleaned)
            
            if anomalies_found > 0:
                st.info(f"Detected and handled {anomalies_found} anomalous values")
            
            return df_cleaned
            
        except Exception as e:
            st.warning(f"Error in anomaly detection: {str(e)}")
            return df
    
    def _detect_outliers_iqr(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Detect outliers using IQR method and cap them"""
        try:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them (more conservative)
            df.loc[df[column] < lower_bound, column] = lower_bound
            df.loc[df[column] > upper_bound, column] = upper_bound
            
            return df
            
        except Exception:
            return df
    
    def mask_pii(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and mask personally identifiable information"""
        try:
            df_cleaned = df.copy()
            pii_found = False
            
            for col in df_cleaned.columns:
                if df_cleaned[col].dtype == 'object':
                    # Check each PII pattern
                    for pii_type, pattern in self.pii_patterns.items():
                        if self._column_contains_pii(df_cleaned[col], pattern):
                            df_cleaned[col] = self._mask_pii_column(df_cleaned[col], pii_type, pattern)
                            pii_found = True
                            st.info(f"Masked {pii_type} data in column '{col}'")
            
            return df_cleaned
            
        except Exception as e:
            st.error(f"Error masking PII: {str(e)}")
            return df
    
    def _column_contains_pii(self, series: pd.Series, pattern: str) -> bool:
        """Check if a column contains PII based on pattern matching"""
        try:
            sample_size = min(100, len(series))
            sample = series.dropna().head(sample_size)
            
            matches = 0
            for value in sample:
                if re.search(pattern, str(value)):
                    matches += 1
            
            # If more than 10% of sample matches, consider it PII
            return matches / len(sample) > 0.1 if len(sample) > 0 else False
            
        except Exception:
            return False
    
    def _mask_pii_column(self, series: pd.Series, pii_type: str, pattern: str) -> pd.Series:
        """Mask PII in a column while preserving data utility"""
        def mask_value(value):
            if pd.isna(value):
                return value
            
            value_str = str(value)
            
            if pii_type == 'email':
                # Keep domain for analytics, mask username
                if '@' in value_str:
                    username, domain = value_str.split('@', 1)
                    masked_username = username[0] + '*' * (len(username) - 1) if len(username) > 1 else '*'
                    return f"{masked_username}@{domain}"
            
            elif pii_type == 'phone':
                # Keep last 4 digits
                digits = re.findall(r'\d', value_str)
                if len(digits) >= 4:
                    return 'XXX-XXX-' + ''.join(digits[-4:])
            
            elif pii_type == 'ssn':
                # Keep last 4 digits
                digits = re.findall(r'\d', value_str)
                if len(digits) >= 4:
                    return 'XXX-XX-' + ''.join(digits[-4:])
            
            elif pii_type == 'credit_card':
                # Keep last 4 digits
                digits = re.findall(r'\d', value_str)
                if len(digits) >= 4:
                    return 'XXXX-XXXX-XXXX-' + ''.join(digits[-4:])
            
            # Default masking
            return 'MASKED_' + pii_type.upper()
        
        return series.apply(mask_value)
    
    def combine_datasets(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Intelligently combine multiple datasets"""
        try:
            if len(dataframes) == 1:
                return dataframes[0]
            
            # Find common columns across all datasets
            common_columns = set(dataframes[0].columns)
            for df in dataframes[1:]:
                common_columns = common_columns.intersection(set(df.columns))
            
            if common_columns:
                # Use only common columns for concatenation
                aligned_dfs = [df[list(common_columns)] for df in dataframes]
                combined = pd.concat(aligned_dfs, ignore_index=True, sort=False)
            else:
                # No common columns - concatenate with all columns
                combined = pd.concat(dataframes, ignore_index=True, sort=False)
            
            # Remove duplicates from combined dataset
            combined = self.remove_duplicates(combined)
            
            return combined
            
        except Exception as e:
            st.error(f"Error combining datasets: {str(e)}")
            return dataframes[0] if dataframes else pd.DataFrame()
    
    def get_quality_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate data quality metrics"""
        try:
            total_cells = df.shape[0] * df.shape[1]
            
            # Completeness: percentage of non-null values
            non_null_cells = df.count().sum()
            completeness = non_null_cells / total_cells if total_cells > 0 else 0
            
            # Uniqueness: average uniqueness across columns
            uniqueness_scores = []
            for col in df.columns:
                if len(df) > 0:
                    unique_ratio = df[col].nunique() / len(df)
                    uniqueness_scores.append(unique_ratio)
            
            uniqueness = np.mean(uniqueness_scores) if uniqueness_scores else 0
            
            # Validity: percentage of values that seem valid (simplified)
            validity_scores = []
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    # For numeric columns, check for infinite values
                    valid_count = np.isfinite(df[col]).sum()
                    validity_scores.append(valid_count / len(df) if len(df) > 0 else 0)
                else:
                    # For non-numeric, check for empty strings
                    valid_count = (df[col].astype(str).str.strip() != '').sum()
                    validity_scores.append(valid_count / len(df) if len(df) > 0 else 0)
            
            validity = np.mean(validity_scores) if validity_scores else 0
            
            return {
                'completeness': completeness,
                'uniqueness': uniqueness,
                'validity': validity
            }
            
        except Exception as e:
            st.error(f"Error calculating quality metrics: {str(e)}")
            return {'completeness': 0, 'uniqueness': 0, 'validity': 0}
    
    def normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize schema for SQL database storage"""
        try:
            df_normalized = df.copy()
            
            # Convert column names to SQL-friendly format
            new_columns = []
            for col in df_normalized.columns:
                # Remove special characters, convert to lowercase
                sql_col = re.sub(r'[^\w]', '_', str(col).lower())
                sql_col = re.sub(r'_+', '_', sql_col).strip('_')
                
                # Ensure it doesn't start with a number
                if sql_col and sql_col[0].isdigit():
                    sql_col = f"col_{sql_col}"
                
                # Ensure it's not empty
                if not sql_col:
                    sql_col = f"column_{len(new_columns)}"
                
                new_columns.append(sql_col)
            
            df_normalized.columns = new_columns
            
            # Handle data types for SQL compatibility
            for col in df_normalized.columns:
                if df_normalized[col].dtype == 'object':
                    # Try to convert to more specific types
                    df_normalized[col] = self._optimize_object_column(df_normalized[col])
            
            return df_normalized
            
        except Exception as e:
            st.error(f"Error normalizing schema: {str(e)}")
            return df
    
    def _optimize_object_column(self, series: pd.Series) -> pd.Series:
        """Optimize object columns by trying to convert to more specific types"""
        try:
            # Try datetime conversion
            try:
                return pd.to_datetime(series, errors='ignore')
            except:
                pass
            
            # Try numeric conversion
            try:
                return pd.to_numeric(series, errors='ignore')
            except:
                pass
            
            # Keep as string but ensure consistency
            return series.astype(str)
            
        except Exception:
            return series
