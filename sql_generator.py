import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Tuple, Optional
import streamlit as st
from datetime import datetime

class SQLGenerator:
    """Generate SQL schemas and statements from pandas DataFrames"""
    
    def __init__(self):
        self.sql_type_mapping = {
            'int64': 'BIGINT',
            'int32': 'INTEGER',
            'int16': 'SMALLINT',
            'int8': 'SMALLINT',
            'float64': 'DOUBLE PRECISION',
            'float32': 'REAL',
            'object': 'TEXT',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP',
            'datetime64[ns, UTC]': 'TIMESTAMP WITH TIME ZONE',
            'category': 'TEXT'
        }
        
        self.reserved_words = {
            'user', 'order', 'group', 'select', 'from', 'where', 'insert', 'update', 
            'delete', 'create', 'drop', 'table', 'index', 'primary', 'key', 'foreign',
            'references', 'constraint', 'check', 'unique', 'not', 'null', 'default',
            'distinct', 'count', 'sum', 'avg', 'max', 'min', 'join', 'inner', 'outer',
            'left', 'right', 'on', 'as', 'and', 'or', 'in', 'like', 'between', 'exists'
        }
    
    def generate_schema(self, df: pd.DataFrame, table_name: str, include_indexes: bool = True) -> Dict[str, Any]:
        """Generate complete SQL schema from DataFrame"""
        try:
            # Clean table name
            clean_table_name = self._clean_identifier(table_name)
            
            # Analyze data types and constraints
            column_mapping = self._analyze_columns(df)
            
            # Generate CREATE TABLE statement
            create_table = self._generate_create_table(clean_table_name, column_mapping)
            
            # Generate sample INSERT statements
            sample_inserts = self._generate_sample_inserts(clean_table_name, df, column_mapping, limit=5)
            
            # Generate full INSERT statements
            full_inserts = self._generate_full_inserts(clean_table_name, df, column_mapping)
            
            # Generate indexes if requested
            indexes = []
            if include_indexes:
                indexes = self._generate_indexes(clean_table_name, df, column_mapping)
            
            # Identify potential primary keys
            primary_keys = self._identify_primary_keys(df, column_mapping)
            
            return {
                'table_name': clean_table_name,
                'create_table': create_table,
                'sample_inserts': sample_inserts,
                'full_inserts': full_inserts,
                'indexes': indexes,
                'column_mapping': column_mapping,
                'primary_keys': primary_keys,
                'row_count': len(df)
            }
            
        except Exception as e:
            st.error(f"Error generating SQL schema: {str(e)}")
            return {}
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze each column to determine SQL data types and constraints"""
        column_mapping = {}
        
        for col in df.columns:
            clean_col_name = self._clean_identifier(col)
            
            # Basic type mapping
            pandas_type = str(df[col].dtype)
            sql_type = self.sql_type_mapping.get(pandas_type, 'TEXT')
            
            # Analyze column for better type determination
            sql_type, constraints = self._analyze_column_details(df[col], sql_type)
            
            # Check for nullability
            has_nulls = df[col].isnull().any()
            nullable = has_nulls
            
            # Check for uniqueness
            is_unique = df[col].nunique() == len(df.dropna(subset=[col]))
            
            column_mapping[col] = {
                'original_name': col,
                'sql_name': clean_col_name,
                'pandas_type': pandas_type,
                'sql_type': sql_type,
                'nullable': nullable,
                'unique': is_unique,
                'constraints': constraints,
                'sample_values': self._get_sample_values(df[col]),
                'comments': self._generate_column_comment(df[col])
            }
        
        return column_mapping
    
    def _analyze_column_details(self, series: pd.Series, base_sql_type: str) -> Tuple[str, List[str]]:
        """Analyze column details to refine SQL type and add constraints"""
        constraints = []
        sql_type = base_sql_type
        
        # For text columns, try to determine better types
        if base_sql_type == 'TEXT' and series.dtype == 'object':
            sql_type, text_constraints = self._analyze_text_column(series)
            constraints.extend(text_constraints)
        
        # For numeric columns, check ranges
        elif base_sql_type in ['BIGINT', 'INTEGER', 'SMALLINT']:
            sql_type, numeric_constraints = self._analyze_numeric_column(series)
            constraints.extend(numeric_constraints)
        
        # Check for positive values
        if series.dtype in ['int64', 'int32', 'int16', 'float64', 'float32']:
            if (series.dropna() >= 0).all():
                constraints.append('CHECK ({} >= 0)'.format(self._clean_identifier(series.name)))
        
        return sql_type, constraints
    
    def _analyze_text_column(self, series: pd.Series) -> Tuple[str, List[str]]:
        """Analyze text column to determine optimal SQL type"""
        constraints = []
        
        # Calculate string lengths
        lengths = series.astype(str).str.len()
        max_length = lengths.max()
        avg_length = lengths.mean()
        
        # Determine if it's a fixed-length field
        unique_lengths = lengths.nunique()
        
        if unique_lengths == 1 and max_length <= 10:
            # Fixed length, short string
            sql_type = f'CHAR({max_length})'
        elif max_length <= 255 and avg_length <= 50:
            # Variable length, reasonable size
            suggested_length = min(255, int(max_length * 1.2))  # Add 20% buffer
            sql_type = f'VARCHAR({suggested_length})'
        elif max_length <= 1000:
            # Medium text
            sql_type = f'VARCHAR({int(max_length * 1.2)})'
        else:
            # Large text
            sql_type = 'TEXT'
        
        # Check for common patterns
        if self._looks_like_email(series):
            constraints.append('CHECK ({} ~* \'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{{2,}}$\')'.format(
                self._clean_identifier(series.name)))
        
        return sql_type, constraints
    
    def _analyze_numeric_column(self, series: pd.Series) -> Tuple[str, List[str]]:
        """Analyze numeric column to optimize SQL type"""
        constraints = []
        
        if series.dtype in ['int64', 'int32', 'int16', 'int8']:
            min_val = series.min()
            max_val = series.max()
            
            # Optimize integer type based on range
            if min_val >= -32768 and max_val <= 32767:
                sql_type = 'SMALLINT'
            elif min_val >= -2147483648 and max_val <= 2147483647:
                sql_type = 'INTEGER'
            else:
                sql_type = 'BIGINT'
        else:
            # Keep original float type
            sql_type = 'DOUBLE PRECISION' if series.dtype == 'float64' else 'REAL'
        
        return sql_type, constraints
    
    def _looks_like_email(self, series: pd.Series) -> bool:
        """Check if a series looks like email addresses"""
        sample_size = min(100, len(series.dropna()))
        if sample_size == 0:
            return False
        
        sample = series.dropna().head(sample_size)
        email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'
        
        matches = sum(1 for value in sample if re.match(email_pattern, str(value)))
        return matches / sample_size > 0.8
    
    def _clean_identifier(self, identifier: str) -> str:
        """Clean and validate SQL identifier"""
        # Convert to string and lowercase
        clean_id = str(identifier).lower()
        
        # Replace spaces and special characters with underscores
        clean_id = re.sub(r'[^\w]', '_', clean_id)
        
        # Remove multiple consecutive underscores
        clean_id = re.sub(r'_+', '_', clean_id)
        
        # Remove leading/trailing underscores
        clean_id = clean_id.strip('_')
        
        # Ensure it doesn't start with a number
        if clean_id and clean_id[0].isdigit():
            clean_id = f'col_{clean_id}'
        
        # Handle empty or reserved words
        if not clean_id or clean_id in self.reserved_words:
            clean_id = f'column_{hash(identifier) % 1000}'
        
        return clean_id
    
    def _get_sample_values(self, series: pd.Series, limit: int = 3) -> List[Any]:
        """Get sample values from a series"""
        try:
            non_null_values = series.dropna()
            if len(non_null_values) == 0:
                return []
            
            # Get unique values
            unique_values = non_null_values.unique()
            sample_values = unique_values[:limit]
            
            # Convert to Python native types for JSON serialization
            return [self._convert_to_native_type(val) for val in sample_values]
            
        except Exception:
            return []
    
    def _convert_to_native_type(self, value):
        """Convert pandas/numpy types to Python native types"""
        if pd.isna(value):
            return None
        elif isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, (pd.Timestamp, np.datetime64)):
            return str(value)
        else:
            return str(value)
    
    def _generate_column_comment(self, series: pd.Series) -> str:
        """Generate descriptive comment for a column"""
        try:
            non_null_count = series.count()
            total_count = len(series)
            unique_count = series.nunique()
            
            comment_parts = []
            
            if non_null_count < total_count:
                null_percentage = ((total_count - non_null_count) / total_count) * 100
                comment_parts.append(f"{null_percentage:.1f}% null")
            
            if unique_count == total_count:
                comment_parts.append("unique values")
            elif unique_count == 1:
                comment_parts.append("constant value")
            else:
                uniqueness_ratio = unique_count / total_count
                if uniqueness_ratio < 0.1:
                    comment_parts.append("low cardinality")
                elif uniqueness_ratio > 0.9:
                    comment_parts.append("high cardinality")
            
            return "; ".join(comment_parts) if comment_parts else ""
            
        except Exception:
            return ""
    
    def _generate_create_table(self, table_name: str, column_mapping: Dict[str, Dict[str, Any]]) -> str:
        """Generate CREATE TABLE statement"""
        try:
            lines = [f'CREATE TABLE {table_name} (']
            
            column_definitions = []
            
            for col_info in column_mapping.values():
                sql_name = col_info['sql_name']
                sql_type = col_info['sql_type']
                nullable = col_info['nullable']
                unique = col_info['unique']
                
                definition = f'    {sql_name} {sql_type}'
                
                if not nullable:
                    definition += ' NOT NULL'
                
                if unique and not nullable:
                    definition += ' UNIQUE'
                
                column_definitions.append(definition)
            
            lines.append(',\n'.join(column_definitions))
            lines.append(');')
            
            # Add comments
            comment_lines = []
            for col_info in column_mapping.values():
                if col_info['comments']:
                    comment_lines.append(
                        f"COMMENT ON COLUMN {table_name}.{col_info['sql_name']} IS '{col_info['comments']}';"
                    )
            
            create_table_sql = '\n'.join(lines)
            
            if comment_lines:
                create_table_sql += '\n\n-- Column comments\n' + '\n'.join(comment_lines)
            
            return create_table_sql
            
        except Exception as e:
            return f"-- Error generating CREATE TABLE: {str(e)}"
    
    def _generate_sample_inserts(self, table_name: str, df: pd.DataFrame, 
                                column_mapping: Dict[str, Dict[str, Any]], limit: int = 5) -> str:
        """Generate sample INSERT statements"""
        try:
            if len(df) == 0:
                return f"-- No data available for {table_name}"
            
            sample_df = df.head(limit)
            return self._generate_inserts_for_dataframe(table_name, sample_df, column_mapping, is_sample=True)
            
        except Exception as e:
            return f"-- Error generating sample INSERTs: {str(e)}"
    
    def _generate_full_inserts(self, table_name: str, df: pd.DataFrame, 
                              column_mapping: Dict[str, Dict[str, Any]]) -> str:
        """Generate INSERT statements for entire dataset"""
        try:
            if len(df) == 0:
                return f"-- No data available for {table_name}"
            
            # Limit to reasonable size for performance
            max_rows = 10000
            if len(df) > max_rows:
                st.warning(f"Dataset has {len(df)} rows. Generating INSERTs for first {max_rows} rows only.")
                df_to_process = df.head(max_rows)
            else:
                df_to_process = df
            
            return self._generate_inserts_for_dataframe(table_name, df_to_process, column_mapping, is_sample=False)
            
        except Exception as e:
            return f"-- Error generating full INSERTs: {str(e)}"
    
    def _generate_inserts_for_dataframe(self, table_name: str, df: pd.DataFrame, 
                                       column_mapping: Dict[str, Dict[str, Any]], is_sample: bool = False) -> str:
        """Generate INSERT statements for a DataFrame"""
        try:
            lines = []
            
            # Add header comment
            if is_sample:
                lines.append(f"-- Sample INSERT statements for {table_name}")
            else:
                lines.append(f"-- INSERT statements for {table_name} ({len(df)} rows)")
            
            # Get column names in order
            sql_columns = [column_mapping[col]['sql_name'] for col in df.columns]
            columns_clause = ', '.join(sql_columns)
            
            # Generate INSERT statements (batch them for efficiency)
            batch_size = 100 if not is_sample else 1
            
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i + batch_size]
                
                if len(batch_df) == 1:
                    # Single row INSERT
                    values = self._format_row_values(batch_df.iloc[0], df.columns, column_mapping)
                    lines.append(f"INSERT INTO {table_name} ({columns_clause}) VALUES ({values});")
                else:
                    # Multi-row INSERT
                    lines.append(f"INSERT INTO {table_name} ({columns_clause}) VALUES")
                    
                    value_lines = []
                    for _, row in batch_df.iterrows():
                        values = self._format_row_values(row, df.columns, column_mapping)
                        value_lines.append(f"    ({values})")
                    
                    lines.append(',\n'.join(value_lines) + ';')
                
                lines.append('')  # Empty line between batches
            
            return '\n'.join(lines)
            
        except Exception as e:
            return f"-- Error generating INSERT statements: {str(e)}"
    
    def _format_row_values(self, row: pd.Series, original_columns: List[str], 
                          column_mapping: Dict[str, Dict[str, Any]]) -> str:
        """Format a row's values for SQL INSERT"""
        formatted_values = []
        
        for col in original_columns:
            value = row[col]
            col_info = column_mapping[col]
            sql_type = col_info['sql_type']
            
            formatted_value = self._format_sql_value(value, sql_type)
            formatted_values.append(formatted_value)
        
        return ', '.join(formatted_values)
    
    def _format_sql_value(self, value, sql_type: str) -> str:
        """Format a single value for SQL"""
        if pd.isna(value):
            return 'NULL'
        
        if sql_type in ['TEXT', 'VARCHAR', 'CHAR'] or sql_type.startswith('VARCHAR'):
            # String types - escape quotes
            escaped_value = str(value).replace("'", "''")
            return f"'{escaped_value}'"
        
        elif sql_type in ['BIGINT', 'INTEGER', 'SMALLINT']:
            # Integer types
            return str(int(value))
        
        elif sql_type in ['DOUBLE PRECISION', 'REAL']:
            # Float types
            return str(float(value))
        
        elif sql_type == 'BOOLEAN':
            # Boolean type
            return 'TRUE' if bool(value) else 'FALSE'
        
        elif sql_type.startswith('TIMESTAMP'):
            # Timestamp types
            if isinstance(value, (pd.Timestamp, datetime)):
                return f"'{value.isoformat()}'"
            else:
                return f"'{str(value)}'"
        
        else:
            # Default: treat as string
            escaped_value = str(value).replace("'", "''")
            return f"'{escaped_value}'"
    
    def _generate_indexes(self, table_name: str, df: pd.DataFrame, 
                         column_mapping: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate useful indexes for the table"""
        try:
            indexes = []
            
            # Create indexes for unique columns
            for col_info in column_mapping.values():
                if col_info['unique'] and not col_info['nullable']:
                    sql_name = col_info['sql_name']
                    index_name = f"idx_{table_name}_{sql_name}"
                    indexes.append(f"CREATE UNIQUE INDEX {index_name} ON {table_name} ({sql_name});")
            
            # Create indexes for high-cardinality non-unique columns
            for col, col_info in column_mapping.items():
                if not col_info['unique']:
                    cardinality_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                    
                    # Index columns with moderate to high cardinality
                    if 0.1 < cardinality_ratio < 0.9:
                        sql_name = col_info['sql_name']
                        index_name = f"idx_{table_name}_{sql_name}"
                        indexes.append(f"CREATE INDEX {index_name} ON {table_name} ({sql_name});")
            
            return indexes
            
        except Exception as e:
            st.warning(f"Could not generate indexes: {str(e)}")
            return []
    
    def _identify_primary_keys(self, df: pd.DataFrame, 
                              column_mapping: Dict[str, Dict[str, Any]]) -> List[str]:
        """Identify potential primary key columns"""
        try:
            primary_key_candidates = []
            
            for col, col_info in column_mapping.items():
                # Primary key candidates: unique, non-null columns
                if col_info['unique'] and not col_info['nullable']:
                    primary_key_candidates.append(col_info['sql_name'])
            
            return primary_key_candidates
            
        except Exception:
            return []
    
    def export_schema_script(self, schema_info: Dict[str, Any]) -> str:
        """Generate complete SQL script with schema, data, and indexes"""
        try:
            script_parts = []
            
            # Header
            script_parts.append("-- Enterprise Data Analysis Platform")
            script_parts.append("-- Generated SQL Schema and Data")
            script_parts.append(f"-- Generated on: {datetime.now().isoformat()}")
            script_parts.append(f"-- Table: {schema_info.get('table_name', 'unknown')}")
            script_parts.append(f"-- Rows: {schema_info.get('row_count', 0):,}")
            script_parts.append("")
            
            # CREATE TABLE
            script_parts.append("-- Create table structure")
            script_parts.append(schema_info.get('create_table', ''))
            script_parts.append("")
            
            # Indexes
            if schema_info.get('indexes'):
                script_parts.append("-- Create indexes")
                script_parts.extend(schema_info['indexes'])
                script_parts.append("")
            
            # Data
            script_parts.append("-- Insert data")
            script_parts.append(schema_info.get('full_inserts', ''))
            
            return '\n'.join(script_parts)
            
        except Exception as e:
            return f"-- Error generating complete script: {str(e)}"
