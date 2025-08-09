import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.stats as stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """Comprehensive data analysis with statistical profiling and business insights"""
    
    def __init__(self):
        self.analysis_cache = {}
        
    def comprehensive_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive analysis of the dataset"""
        try:
            analysis_results = {
                'statistical_profile': self._statistical_profile(df),
                'correlations': self._correlation_analysis(df),
                'anomalies': self._anomaly_detection(df),
                'trends': self._trend_analysis(df),
                'predictions': self._predictive_modeling(df),
                'clustering': self._clustering_analysis(df),
                'quality_assessment': self._data_quality_assessment(df)
            }
            
            return analysis_results
            
        except Exception as e:
            st.error(f"Error in comprehensive analysis: {str(e)}")
            return {}
    
    def _statistical_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical profile for all columns"""
        try:
            profile = {
                'dataset_info': {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                    'duplicate_rows': df.duplicated().sum()
                },
                'column_profiles': {}
            }
            
            for col in df.columns:
                col_profile = self._analyze_column(df[col])
                profile['column_profiles'][col] = col_profile
            
            return profile
            
        except Exception as e:
            st.error(f"Error in statistical profiling: {str(e)}")
            return {}
    
    def _analyze_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze a single column comprehensively"""
        try:
            profile = {
                'data_type': str(series.dtype),
                'non_null_count': series.count(),
                'null_count': series.isnull().sum(),
                'null_percentage': (series.isnull().sum() / len(series)) * 100,
                'unique_count': series.nunique(),
                'unique_percentage': (series.nunique() / len(series)) * 100 if len(series) > 0 else 0
            }
            
            if series.dtype in ['int64', 'float64']:
                # Numeric column analysis
                profile.update({
                    'min': float(series.min()) if not series.empty else None,
                    'max': float(series.max()) if not series.empty else None,
                    'mean': float(series.mean()) if not series.empty else None,
                    'median': float(series.median()) if not series.empty else None,
                    'std': float(series.std()) if not series.empty else None,
                    'skewness': float(series.skew()) if not series.empty else None,
                    'kurtosis': float(series.kurtosis()) if not series.empty else None,
                    'quartile_25': float(series.quantile(0.25)) if not series.empty else None,
                    'quartile_75': float(series.quantile(0.75)) if not series.empty else None,
                    'outliers_count': self._count_outliers(series),
                    'zero_count': (series == 0).sum(),
                    'negative_count': (series < 0).sum()
                })
                
                # Distribution analysis
                profile['distribution'] = self._analyze_distribution(series)
                
            elif series.dtype == 'object':
                # String column analysis
                profile.update({
                    'most_frequent': series.mode().iloc[0] if not series.mode().empty else None,
                    'most_frequent_count': series.value_counts().iloc[0] if not series.empty else 0,
                    'average_length': series.astype(str).str.len().mean() if not series.empty else 0,
                    'min_length': series.astype(str).str.len().min() if not series.empty else 0,
                    'max_length': series.astype(str).str.len().max() if not series.empty else 0,
                    'empty_strings': (series == '').sum(),
                    'whitespace_only': series.astype(str).str.strip().eq('').sum()
                })
                
                # Pattern analysis
                profile['patterns'] = self._analyze_patterns(series)
                
            elif 'datetime' in str(series.dtype):
                # DateTime column analysis
                profile.update({
                    'min_date': str(series.min()) if not series.empty else None,
                    'max_date': str(series.max()) if not series.empty else None,
                    'date_range_days': (series.max() - series.min()).days if not series.empty else 0
                })
            
            return profile
            
        except Exception as e:
            return {'error': str(e)}
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method"""
        try:
            if series.dtype not in ['int64', 'float64'] or len(series) < 4:
                return 0
            
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            return len(outliers)
            
        except Exception:
            return 0
    
    def _analyze_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze the distribution of numeric data"""
        try:
            if len(series.dropna()) < 10:
                return {'distribution_type': 'insufficient_data'}
            
            # Test for normality
            _, p_value_normality = stats.normaltest(series.dropna())
            is_normal = p_value_normality > 0.05
            
            # Test for uniformity
            _, p_value_uniform = stats.kstest(series.dropna(), 'uniform')
            is_uniform = p_value_uniform > 0.05
            
            return {
                'is_normal': is_normal,
                'normality_p_value': float(p_value_normality),
                'is_uniform': is_uniform,
                'uniformity_p_value': float(p_value_uniform),
                'distribution_type': self._infer_distribution_type(series)
            }
            
        except Exception:
            return {'distribution_type': 'unknown'}
    
    def _infer_distribution_type(self, series: pd.Series) -> str:
        """Infer the most likely distribution type"""
        try:
            clean_series = series.dropna()
            if len(clean_series) < 10:
                return 'insufficient_data'
            
            # Simple heuristics for common distributions
            skewness = clean_series.skew()
            kurtosis = clean_series.kurtosis()
            
            if abs(skewness) < 0.5 and abs(kurtosis) < 3:
                return 'approximately_normal'
            elif skewness > 1:
                return 'right_skewed'
            elif skewness < -1:
                return 'left_skewed'
            elif kurtosis > 3:
                return 'heavy_tailed'
            else:
                return 'unknown'
                
        except Exception:
            return 'unknown'
    
    def _analyze_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze patterns in string data"""
        try:
            patterns = {
                'contains_numbers': series.astype(str).str.contains(r'\d').sum(),
                'contains_special_chars': series.astype(str).str.contains(r'[^\w\s]').sum(),
                'all_uppercase': series.astype(str).str.isupper().sum(),
                'all_lowercase': series.astype(str).str.islower().sum(),
                'starts_with_number': series.astype(str).str.match(r'^\d').sum(),
                'email_like': series.astype(str).str.contains(r'@.*\.').sum(),
                'url_like': series.astype(str).str.contains(r'http[s]?://').sum(),
                'phone_like': series.astype(str).str.contains(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}').sum()
            }
            
            return patterns
            
        except Exception:
            return {}
    
    def _correlation_analysis(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Analyze correlations between numeric columns"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) < 2:
                return None
            
            correlation_matrix = numeric_df.corr()
            
            return correlation_matrix
            
        except Exception as e:
            st.warning(f"Could not perform correlation analysis: {str(e)}")
            return None
    
    def _anomaly_detection(self, df: pd.DataFrame) -> List[str]:
        """Detect anomalies in the dataset"""
        try:
            anomalies = []
            
            # Numeric anomalies
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if len(df[col].dropna()) > 10:
                    # Statistical anomalies
                    outlier_count = self._count_outliers(df[col])
                    if outlier_count > len(df) * 0.05:  # More than 5% outliers
                        anomalies.append(f"Column '{col}' has {outlier_count} outliers ({outlier_count/len(df)*100:.1f}%)")
                    
                    # Check for suspicious patterns
                    if df[col].nunique() == 1:
                        anomalies.append(f"Column '{col}' has only one unique value")
                    
                    # Check for extreme skewness
                    skewness = abs(df[col].skew())
                    if skewness > 3:
                        anomalies.append(f"Column '{col}' is highly skewed (skewness: {skewness:.2f})")
            
            # Text anomalies
            text_cols = df.select_dtypes(include=['object']).columns
            
            for col in text_cols:
                # Check for inconsistent formatting
                if df[col].dtype == 'object':
                    unique_lengths = df[col].astype(str).str.len().nunique()
                    if unique_lengths > len(df) * 0.8:  # Too many different lengths
                        anomalies.append(f"Column '{col}' has inconsistent text lengths")
                    
                    # Check for mixed case issues
                    upper_count = df[col].astype(str).str.isupper().sum()
                    lower_count = df[col].astype(str).str.islower().sum()
                    mixed_ratio = (len(df) - upper_count - lower_count) / len(df)
                    
                    if mixed_ratio > 0.5:
                        anomalies.append(f"Column '{col}' has mixed case formatting")
            
            return anomalies
            
        except Exception as e:
            st.warning(f"Could not perform anomaly detection: {str(e)}")
            return []
    
    def _trend_analysis(self, df: pd.DataFrame) -> List[str]:
        """Analyze trends in the data"""
        try:
            trends = []
            
            # Look for datetime columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(datetime_cols) > 0 and len(numeric_cols) > 0:
                date_col = datetime_cols[0]
                
                for num_col in numeric_cols:
                    try:
                        # Sort by date and analyze trend
                        sorted_df = df.sort_values(date_col)
                        
                        # Simple linear trend analysis
                        x = range(len(sorted_df))
                        y = sorted_df[num_col].values
                        
                        # Remove NaN values
                        mask = ~np.isnan(y)
                        if np.sum(mask) > 2:
                            correlation = np.corrcoef(np.array(x)[mask], y[mask])[0, 1]
                            
                            if abs(correlation) > 0.7:
                                trend_direction = "increasing" if correlation > 0 else "decreasing"
                                trends.append(f"Strong {trend_direction} trend in '{num_col}' over time (correlation: {correlation:.3f})")
                            elif abs(correlation) > 0.3:
                                trend_direction = "increasing" if correlation > 0 else "decreasing"
                                trends.append(f"Moderate {trend_direction} trend in '{num_col}' over time (correlation: {correlation:.3f})")
                    
                    except Exception:
                        continue
            
            # Analyze correlations for business insights
            correlations = self._correlation_analysis(df)
            if correlations is not None:
                # Find strong correlations
                for i in range(len(correlations.columns)):
                    for j in range(i + 1, len(correlations.columns)):
                        correlation_value = correlations.iloc[i, j]
                        if abs(correlation_value) > 0.7:
                            col1, col2 = correlations.columns[i], correlations.columns[j]
                            relationship = "positively" if correlation_value > 0 else "negatively"
                            trends.append(f"Strong {relationship} correlated variables: '{col1}' and '{col2}' (r={correlation_value:.3f})")
            
            return trends
            
        except Exception as e:
            st.warning(f"Could not perform trend analysis: {str(e)}")
            return []
    
    def _predictive_modeling(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build simple predictive models for numeric target variables"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return {'error': 'Insufficient numeric columns for modeling'}
            
            predictions = {}
            
            # Try to build models for each numeric column as target
            for target_col in numeric_cols[:3]:  # Limit to first 3 for performance
                try:
                    # Prepare features (all other numeric columns)
                    feature_cols = [col for col in numeric_cols if col != target_col]
                    
                    if len(feature_cols) == 0:
                        continue
                    
                    # Prepare data
                    X = df[feature_cols].fillna(df[feature_cols].median())
                    y = df[target_col].fillna(df[target_col].median())
                    
                    # Remove rows where target is NaN
                    valid_mask = ~pd.isna(y)
                    X = X[valid_mask]
                    y = y[valid_mask]
                    
                    if len(X) < 10:
                        continue
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    
                    # Build model
                    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    
                    # Feature importance
                    feature_importance = dict(zip(feature_cols, model.feature_importances_))
                    
                    predictions[target_col] = {
                        'r2_score': float(r2),
                        'rmse': float(rmse),
                        'feature_importance': feature_importance,
                        'model_type': 'RandomForest'
                    }
                
                except Exception:
                    continue
            
            return predictions
            
        except Exception as e:
            st.warning(f"Could not perform predictive modeling: {str(e)}")
            return {}
    
    def _clustering_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis on numeric data"""
        try:
            numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
            
            if len(numeric_df.columns) < 2 or len(numeric_df) < 10:
                return {'error': 'Insufficient data for clustering'}
            
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)
            
            # Determine optimal number of clusters (max 10 for performance)
            max_k = min(10, len(numeric_df) // 2)
            inertias = []
            
            for k in range(2, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(scaled_data)
                inertias.append(kmeans.inertia_)
            
            # Use elbow method to find optimal k
            optimal_k = 3  # Default
            if len(inertias) > 1:
                # Simple elbow detection
                differences = np.diff(inertias)
                if len(differences) > 0:
                    optimal_k = np.argmax(differences) + 2
            
            # Perform final clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            # Analyze clusters
            cluster_summary = {}
            for i in range(optimal_k):
                cluster_mask = cluster_labels == i
                cluster_data = numeric_df[cluster_mask]
                
                cluster_summary[f'cluster_{i}'] = {
                    'size': int(np.sum(cluster_mask)),
                    'percentage': float(np.sum(cluster_mask) / len(numeric_df) * 100),
                    'mean_values': cluster_data.mean().to_dict()
                }
            
            return {
                'optimal_clusters': int(optimal_k),
                'cluster_summary': cluster_summary,
                'silhouette_score': float(self._calculate_silhouette_score(scaled_data, cluster_labels))
            }
            
        except Exception as e:
            st.warning(f"Could not perform clustering analysis: {str(e)}")
            return {}
    
    def _calculate_silhouette_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality"""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(data, labels)
        except Exception:
            return 0.0
    
    def _data_quality_assessment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality"""
        try:
            total_cells = df.shape[0] * df.shape[1]
            
            quality_metrics = {
                'completeness': {
                    'score': float(df.count().sum() / total_cells),
                    'description': 'Percentage of non-null values'
                },
                'uniqueness': {
                    'score': float(np.mean([df[col].nunique() / len(df) for col in df.columns])),
                    'description': 'Average uniqueness ratio across columns'
                },
                'consistency': {
                    'score': self._calculate_consistency_score(df),
                    'description': 'Consistency of data formats and patterns'
                },
                'validity': {
                    'score': self._calculate_validity_score(df),
                    'description': 'Percentage of values that appear valid'
                }
            }
            
            # Overall quality score (weighted average)
            weights = {'completeness': 0.3, 'uniqueness': 0.2, 'consistency': 0.25, 'validity': 0.25}
            overall_score = sum(quality_metrics[metric]['score'] * weights[metric] for metric in weights)
            
            quality_metrics['overall_score'] = {
                'score': float(overall_score),
                'description': 'Weighted average of all quality metrics',
                'grade': self._get_quality_grade(overall_score)
            }
            
            return quality_metrics
            
        except Exception as e:
            st.warning(f"Could not assess data quality: {str(e)}")
            return {}
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Calculate consistency score based on format patterns"""
        try:
            consistency_scores = []
            
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check format consistency for text columns
                    lengths = df[col].astype(str).str.len()
                    length_variance = lengths.var() / lengths.mean() if lengths.mean() > 0 else 0
                    
                    # Lower variance in length indicates better consistency
                    consistency_score = max(0, 1 - length_variance / 10)
                    consistency_scores.append(consistency_score)
                else:
                    # For numeric columns, consistency is generally good
                    consistency_scores.append(0.9)
            
            return float(np.mean(consistency_scores)) if consistency_scores else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_validity_score(self, df: pd.DataFrame) -> float:
        """Calculate validity score based on data type appropriateness"""
        try:
            validity_scores = []
            
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    # For numeric columns, check for infinite values
                    finite_ratio = np.isfinite(df[col]).sum() / len(df)
                    validity_scores.append(finite_ratio)
                else:
                    # For text columns, check for non-empty strings
                    non_empty_ratio = (df[col].astype(str).str.strip() != '').sum() / len(df)
                    validity_scores.append(non_empty_ratio)
            
            return float(np.mean(validity_scores)) if validity_scores else 0.5
            
        except Exception:
            return 0.5
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade"""
        if score >= 0.9:
            return 'A (Excellent)'
        elif score >= 0.8:
            return 'B (Good)'
        elif score >= 0.7:
            return 'C (Fair)'
        elif score >= 0.6:
            return 'D (Poor)'
        else:
            return 'F (Very Poor)'
    
    def generate_business_insights(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate business-relevant insights from the analysis"""
        try:
            insights = []
            
            # Data volume insights
            row_count = len(df)
            if row_count > 100000:
                insights.append(f"Large dataset with {row_count:,} records - suitable for robust statistical analysis and machine learning")
            elif row_count < 1000:
                insights.append(f"Small dataset with {row_count:,} records - statistical conclusions may have limited confidence")
            
            # Quality insights
            if 'quality_assessment' in analysis_results:
                quality = analysis_results['quality_assessment']
                if 'overall_score' in quality:
                    score = quality['overall_score']['score']
                    grade = quality['overall_score']['grade']
                    insights.append(f"Data quality assessment: {grade} (score: {score:.2f})")
                    
                    if score < 0.7:
                        insights.append("⚠️ Data quality issues detected - recommend additional cleaning before analysis")
            
            # Correlation insights
            if 'correlations' in analysis_results and analysis_results['correlations'] is not None:
                corr_matrix = analysis_results['correlations']
                strong_correlations = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        correlation = corr_matrix.iloc[i, j]
                        if abs(correlation) > 0.7:
                            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                            direction = "positive" if correlation > 0 else "negative"
                            strong_correlations.append(f"{col1} and {col2} ({direction}, r={correlation:.3f})")
                
                if strong_correlations:
                    insights.append(f"Strong correlations found: {', '.join(strong_correlations[:3])}")
                    insights.append("💡 These relationships could be leveraged for predictive modeling or business optimization")
            
            # Predictive modeling insights
            if 'predictions' in analysis_results:
                predictions = analysis_results['predictions']
                best_models = [(target, model) for target, model in predictions.items() 
                              if isinstance(model, dict) and model.get('r2_score', 0) > 0.7]
                
                if best_models:
                    target, model = best_models[0]
                    r2 = model['r2_score']
                    insights.append(f"Predictive model for '{target}' shows good performance (R² = {r2:.3f})")
                    
                    # Top features
                    if 'feature_importance' in model:
                        top_features = sorted(model['feature_importance'].items(), 
                                            key=lambda x: x[1], reverse=True)[:3]
                        feature_names = [f[0] for f in top_features]
                        insights.append(f"Key predictive factors for '{target}': {', '.join(feature_names)}")
            
            # Clustering insights
            if 'clustering' in analysis_results:
                clustering = analysis_results['clustering']
                if 'optimal_clusters' in clustering:
                    num_clusters = clustering['optimal_clusters']
                    insights.append(f"Natural data segmentation: {num_clusters} distinct groups identified")
                    
                    if 'cluster_summary' in clustering:
                        largest_cluster = max(clustering['cluster_summary'].items(), 
                                            key=lambda x: x[1]['size'])
                        cluster_name, cluster_info = largest_cluster
                        percentage = cluster_info['percentage']
                        insights.append(f"Largest segment represents {percentage:.1f}% of the data")
            
            # Anomaly insights
            if 'anomalies' in analysis_results and analysis_results['anomalies']:
                anomaly_count = len(analysis_results['anomalies'])
                insights.append(f"⚠️ {anomaly_count} data quality issues identified - review recommended")
            
            # Missing data insights
            missing_data = df.isnull().sum().sum()
            total_cells = df.shape[0] * df.shape[1]
            missing_percentage = (missing_data / total_cells) * 100
            
            if missing_percentage > 10:
                insights.append(f"⚠️ Significant missing data: {missing_percentage:.1f}% of values are null")
            elif missing_percentage > 0:
                insights.append(f"Minor missing data: {missing_percentage:.1f}% of values are null")
            else:
                insights.append("✅ Complete dataset with no missing values")
            
            # Business recommendations
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            text_cols = len(df.select_dtypes(include=['object']).columns)
            
            if numeric_cols > text_cols:
                insights.append("💼 Dataset is primarily quantitative - well-suited for statistical analysis and forecasting")
            elif text_cols > numeric_cols:
                insights.append("💼 Dataset is primarily qualitative - consider text analysis and categorization techniques")
            
            return insights
            
        except Exception as e:
            st.warning(f"Could not generate business insights: {str(e)}")
            return ["Analysis completed successfully - review detailed results above"]
