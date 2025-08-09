import pandas as pd
import json
import xml.etree.ElementTree as ET
import PyPDF2
import xmltodict
import sqlite3
import re
from io import StringIO, BytesIO
import streamlit as st
from typing import Optional, Dict, Any, List
import openpyxl

class DataProcessor:
    """Handles processing of multiple file formats into pandas DataFrames"""
    
    def __init__(self):
        self.supported_formats = [
            'csv', 'xlsx', 'xls', 'json', 'xml', 'pdf', 'sql', 'tsv', 'txt'
        ]
    
    def process_file(self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Process uploaded file based on its type and return a pandas DataFrame
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pandas.DataFrame or None if processing fails
        """
        try:
            file_name = uploaded_file.name.lower()
            file_extension = file_name.split('.')[-1]
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            if file_extension == 'csv':
                return self._process_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                return self._process_excel(uploaded_file)
            elif file_extension == 'json':
                return self._process_json(uploaded_file)
            elif file_extension == 'xml':
                return self._process_xml(uploaded_file)
            elif file_extension == 'pdf':
                return self._process_pdf(uploaded_file)
            elif file_extension == 'sql':
                return self._process_sql(uploaded_file)
            elif file_extension in ['tsv', 'txt']:
                return self._process_tsv(uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None
                
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            return None
    
    def _process_csv(self, file) -> pd.DataFrame:
        """Process CSV files with intelligent delimiter detection"""
        try:
            # Read first few lines to detect delimiter
            file.seek(0)
            sample = file.read(1024).decode('utf-8')
            file.seek(0)
            
            # Try common delimiters
            delimiters = [',', ';', '\t', '|']
            best_delimiter = ','
            max_cols = 0
            
            for delimiter in delimiters:
                cols = len(sample.split('\n')[0].split(delimiter))
                if cols > max_cols:
                    max_cols = cols
                    best_delimiter = delimiter
            
            # Read CSV with detected delimiter
            df = pd.read_csv(file, delimiter=best_delimiter, encoding='utf-8', low_memory=False)
            
            # Handle common encoding issues
            if df.empty:
                file.seek(0)
                df = pd.read_csv(file, delimiter=best_delimiter, encoding='latin-1', low_memory=False)
            
            return df
            
        except Exception as e:
            # Fallback to basic CSV reading
            file.seek(0)
            return pd.read_csv(file, encoding='utf-8', low_memory=False)
    
    def _process_excel(self, file) -> pd.DataFrame:
        """Process Excel files (both .xlsx and .xls)"""
        try:
            # Read all sheets and combine them
            excel_file = pd.ExcelFile(file)
            
            if len(excel_file.sheet_names) == 1:
                # Single sheet
                return pd.read_excel(file, sheet_name=0)
            else:
                # Multiple sheets - combine them
                dfs = []
                for sheet_name in excel_file.sheet_names:
                    try:
                        df = pd.read_excel(file, sheet_name=sheet_name)
                        if not df.empty:
                            df['_source_sheet'] = sheet_name
                            dfs.append(df)
                    except Exception as e:
                        st.warning(f"Could not read sheet '{sheet_name}': {str(e)}")
                        continue
                
                if dfs:
                    return pd.concat(dfs, ignore_index=True, sort=False)
                else:
                    return pd.DataFrame()
                    
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
            return pd.DataFrame()
    
    def _process_json(self, file) -> pd.DataFrame:
        """Process JSON files with flexible structure handling"""
        try:
            file.seek(0)
            content = file.read().decode('utf-8')
            data = json.loads(content)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Array of objects
                return pd.json_normalize(data)
            elif isinstance(data, dict):
                # Check if it's a nested structure
                if any(isinstance(v, list) for v in data.values()):
                    # Find the main data array
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0:
                            return pd.json_normalize(value)
                    
                    # If no list found, normalize the dict itself
                    return pd.json_normalize([data])
                else:
                    # Single object
                    return pd.json_normalize([data])
            else:
                st.error("Unsupported JSON structure")
                return pd.DataFrame()
                
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error processing JSON: {str(e)}")
            return pd.DataFrame()
    
    def _process_xml(self, file) -> pd.DataFrame:
        """Process XML files by converting to JSON then DataFrame"""
        try:
            file.seek(0)
            content = file.read().decode('utf-8')
            
            # Convert XML to dictionary
            data_dict = xmltodict.parse(content)
            
            # Find the main data container
            def extract_records(obj, path=""):
                """Recursively extract list-like structures from XML"""
                records = []
                
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, list):
                            # Found a list - likely our data
                            for item in value:
                                if isinstance(item, dict):
                                    records.append(item)
                        elif isinstance(value, dict):
                            # Recurse into nested dicts
                            records.extend(extract_records(value, f"{path}.{key}" if path else key))
                
                return records
            
            records = extract_records(data_dict)
            
            if records:
                return pd.json_normalize(records)
            else:
                # Fallback: convert entire structure
                return pd.json_normalize([data_dict])
                
        except Exception as e:
            st.error(f"Error processing XML: {str(e)}")
            return pd.DataFrame()
    
    def _process_pdf(self, file) -> pd.DataFrame:
        """Extract tables from PDF files"""
        try:
            file.seek(0)
            reader = PyPDF2.PdfReader(file)
            
            all_text = ""
            for page in reader.pages:
                all_text += page.extract_text() + "\n"
            
            # Try to identify table-like structures
            lines = all_text.split('\n')
            table_lines = []
            
            for line in lines:
                line = line.strip()
                if line and ('\t' in line or '  ' in line or '|' in line):
                    # Looks like tabular data
                    table_lines.append(line)
            
            if not table_lines:
                st.warning("No tabular data found in PDF")
                return pd.DataFrame()
            
            # Try to parse as CSV-like structure
            csv_content = '\n'.join(table_lines)
            
            # Detect delimiter
            if '\t' in csv_content:
                delimiter = '\t'
            elif '|' in csv_content:
                delimiter = '|'
            else:
                # Use multiple spaces as delimiter
                delimiter = r'\s{2,}'  # Two or more spaces
                
                # Convert to comma-separated
                processed_lines = []
                for line in table_lines:
                    processed_lines.append(re.sub(delimiter, ',', line.strip()))
                csv_content = '\n'.join(processed_lines)
                delimiter = ','
            
            # Create DataFrame
            df = pd.read_csv(StringIO(csv_content), delimiter=delimiter)
            return df
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return pd.DataFrame()
    
    def _process_sql(self, file) -> pd.DataFrame:
        """Process SQL dump files"""
        try:
            file.seek(0)
            content = file.read().decode('utf-8')
            
            # Look for INSERT statements
            insert_pattern = r'INSERT INTO\s+(\w+)\s*\([^)]+\)\s*VALUES\s*\([^;]+\);'
            inserts = re.findall(insert_pattern, content, re.IGNORECASE | re.MULTILINE)
            
            if not inserts:
                st.warning("No INSERT statements found in SQL file")
                return pd.DataFrame()
            
            # Extract table structure and data
            # This is a simplified parser - in production, you'd want a proper SQL parser
            
            # For now, try to execute the SQL in a temporary SQLite database
            conn = sqlite3.connect(':memory:')
            
            try:
                # Execute the SQL
                conn.executescript(content)
                
                # Get table names
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                if tables:
                    # Use the first table
                    table_name = tables[0][0]
                    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                    return df
                else:
                    st.warning("No tables found in SQL file")
                    return pd.DataFrame()
                    
            finally:
                conn.close()
                
        except Exception as e:
            st.error(f"Error processing SQL file: {str(e)}")
            return pd.DataFrame()
    
    def _process_tsv(self, file) -> pd.DataFrame:
        """Process TSV (Tab-Separated Values) files"""
        try:
            file.seek(0)
            return pd.read_csv(file, delimiter='\t', encoding='utf-8', low_memory=False)
        except Exception as e:
            st.error(f"Error processing TSV file: {str(e)}")
            return pd.DataFrame()
    
    def get_file_info(self, uploaded_file) -> Dict[str, Any]:
        """Get metadata about the uploaded file"""
        try:
            return {
                'name': uploaded_file.name,
                'size': uploaded_file.size,
                'type': uploaded_file.type,
                'extension': uploaded_file.name.split('.')[-1].lower() if '.' in uploaded_file.name else 'unknown'
            }
        except Exception:
            return {
                'name': 'unknown',
                'size': 0,
                'type': 'unknown',
                'extension': 'unknown'
            }
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate and provide basic info about the processed DataFrame"""
        if df is None or df.empty:
            return {
                'valid': False,
                'error': 'DataFrame is empty or None'
            }
        
        return {
            'valid': True,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum()
        }
