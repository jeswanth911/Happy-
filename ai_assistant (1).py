import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import streamlit as st
from openai import OpenAI
import re
from datetime import datetime

class AIAssistant:
    """AI-powered assistant for data analysis and business insights using OpenRouter API"""
    
    def __init__(self):
        # Using OpenRouter with gpt-4o model for better compatibility and access
        self.model = "openai/gpt-4o"
        
        # Initialize OpenRouter client (compatible with OpenAI SDK)
        api_key = os.getenv("OPENAI_API_KEY")  # OpenRouter API key stored in OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenRouter API key not found in environment variables")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    
    def answer_question(self, df: pd.DataFrame, question: str, analysis_results: Optional[Dict] = None) -> Dict[str, Any]:
        """Answer natural language questions about the dataset"""
        try:
            # Prepare dataset context
            dataset_context = self._prepare_dataset_context(df, analysis_results)
            
            # Determine if question requires SQL query
            needs_query = self._question_needs_query(question)
            
            if needs_query:
                return self._answer_with_query(df, question, dataset_context)
            else:
                return self._answer_with_analysis(df, question, dataset_context, analysis_results)
        
        except Exception as e:
            return {
                'summary': f"I apologize, but I encountered an error while processing your question: {str(e)}",
                'error': str(e)
            }
    
    def _prepare_dataset_context(self, df: pd.DataFrame, analysis_results: Optional[Dict] = None) -> str:
        """Prepare context about the dataset for the AI"""
        context_parts = []
        
        # Basic dataset info
        context_parts.append(f"Dataset has {len(df)} rows and {len(df.columns)} columns.")
        context_parts.append(f"Columns: {', '.join(df.columns)}")
        
        # Data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        if numeric_cols:
            context_parts.append(f"Numeric columns: {', '.join(numeric_cols)}")
        if text_cols:
            context_parts.append(f"Text columns: {', '.join(text_cols)}")
        if datetime_cols:
            context_parts.append(f"Date columns: {', '.join(datetime_cols)}")
        
        # Sample data
        if len(df) > 0:
            context_parts.append(f"Sample data (first 3 rows):\n{df.head(3).to_string()}")
        
        # Analysis results if available
        if analysis_results:
            if 'statistical_profile' in analysis_results:
                context_parts.append("Statistical analysis has been performed on this dataset.")
            if 'correlations' in analysis_results and analysis_results['correlations'] is not None:
                context_parts.append("Correlation analysis is available.")
            if 'trends' in analysis_results and analysis_results['trends']:
                context_parts.append(f"Key trends identified: {'; '.join(analysis_results['trends'][:3])}")
        
        return '\n'.join(context_parts)
    
    def _question_needs_query(self, question: str) -> bool:
        """Determine if the question requires a SQL query to answer"""
        query_keywords = [
            'show me', 'list', 'find', 'get', 'select', 'filter', 'where',
            'top', 'bottom', 'highest', 'lowest', 'maximum', 'minimum',
            'count', 'sum', 'average', 'group by', 'sort', 'order'
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in query_keywords)
    
    def _answer_with_query(self, df: pd.DataFrame, question: str, dataset_context: str) -> Dict[str, Any]:
        """Answer questions that require data querying"""
        try:
            # Generate pandas query code
            query_prompt = f"""
            Given this dataset context:
            {dataset_context}
            
            User question: {question}
            
            Generate Python pandas code to answer this question. Return your response as JSON with these fields:
            - "pandas_code": The pandas code to execute
            - "explanation": Brief explanation of what the code does
            - "summary": Natural language answer to the question
            
            The DataFrame variable name is 'df'. Use appropriate pandas methods like .head(), .tail(), .sort_values(), .groupby(), .agg(), etc.
            Make sure the code is safe to execute and doesn't modify the original DataFrame.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data analysis expert. Generate safe pandas code to query datasets."},
                    {"role": "user", "content": query_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            ai_response = json.loads(content) if content else {}
            
            # Execute the generated code safely
            pandas_code = ai_response.get('pandas_code', '')
            
            if pandas_code:
                # Create a safe execution environment
                safe_globals = {
                    'df': df,
                    'pd': pd,
                    'np': np
                }
                
                try:
                    result = eval(pandas_code, safe_globals)
                    
                    return {
                        'summary': ai_response.get('summary', 'Query executed successfully'),
                        'explanation': ai_response.get('explanation', ''),
                        'pandas_code': pandas_code,
                        'results': result
                    }
                
                except Exception as exec_error:
                    return {
                        'summary': f"I generated a query but encountered an execution error: {str(exec_error)}",
                        'pandas_code': pandas_code,
                        'error': str(exec_error)
                    }
            
            else:
                return {
                    'summary': ai_response.get('summary', 'I was unable to generate a suitable query for your question.'),
                    'explanation': ai_response.get('explanation', '')
                }
        
        except Exception as e:
            return {
                'summary': f"Error generating query response: {str(e)}",
                'error': str(e)
            }
    
    def _answer_with_analysis(self, df: pd.DataFrame, question: str, dataset_context: str, analysis_results: Optional[Dict] = None) -> Dict[str, Any]:
        """Answer analytical questions using existing analysis results"""
        try:
            analysis_context = ""
            if analysis_results:
                # Summarize key analysis findings
                analysis_summary = []
                
                if 'statistical_profile' in analysis_results:
                    analysis_summary.append("Statistical profiling completed")
                
                if 'correlations' in analysis_results and analysis_results['correlations'] is not None:
                    corr_matrix = analysis_results['correlations']
                    # Find strongest correlations
                    correlations_text = self._summarize_correlations(corr_matrix)
                    analysis_summary.append(correlations_text)
                
                if 'trends' in analysis_results and analysis_results['trends']:
                    analysis_summary.append(f"Trends: {'; '.join(analysis_results['trends'][:3])}")
                
                if 'anomalies' in analysis_results and analysis_results['anomalies']:
                    analysis_summary.append(f"Anomalies detected: {len(analysis_results['anomalies'])} issues")
                
                if 'predictions' in analysis_results and analysis_results['predictions']:
                    pred_summary = []
                    for target, model_info in analysis_results['predictions'].items():
                        if isinstance(model_info, dict) and 'r2_score' in model_info:
                            pred_summary.append(f"{target} (R²={model_info['r2_score']:.3f})")
                    if pred_summary:
                        analysis_summary.append(f"Predictive models: {'; '.join(pred_summary[:2])}")
                
                analysis_context = '\n'.join(analysis_summary)
            
            # Generate comprehensive response
            analysis_prompt = f"""
            Dataset Context:
            {dataset_context}
            
            Analysis Results:
            {analysis_context}
            
            User Question: {question}
            
            Provide a comprehensive answer to the user's question based on the dataset and analysis results.
            Return your response as JSON with these fields:
            - "summary": Main answer to the question
            - "insights": List of 2-3 key insights related to the question
            - "recommendations": List of 1-2 actionable recommendations
            
            Focus on business value and actionable insights. Be specific and reference actual data patterns when possible.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior data analyst providing business insights. Be specific, actionable, and focus on business value."},
                    {"role": "user", "content": analysis_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            ai_response = json.loads(response.choices[0].message.content)
            
            return {
                'summary': ai_response.get('summary', 'Analysis completed'),
                'insights': ai_response.get('insights', []),
                'recommendations': ai_response.get('recommendations', [])
            }
        
        except Exception as e:
            return {
                'summary': f"Error generating analytical response: {str(e)}",
                'error': str(e)
            }
    
    def _summarize_correlations(self, corr_matrix: pd.DataFrame) -> str:
        """Summarize the strongest correlations in the matrix"""
        try:
            strong_correlations = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    correlation = corr_matrix.iloc[i, j]
                    if abs(correlation) > 0.7:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        direction = "positive" if correlation > 0 else "negative"
                        strong_correlations.append(f"{col1} ↔ {col2} ({direction}, r={correlation:.3f})")
            
            if strong_correlations:
                return f"Strong correlations: {'; '.join(strong_correlations[:3])}"
            else:
                return "No strong correlations found"
        
        except Exception:
            return "Correlation analysis available"
    
    def generate_executive_summary(self, df: pd.DataFrame, analysis_results: Optional[Dict] = None, processing_options: Optional[Dict] = None) -> str:
        """Generate a comprehensive executive summary for business stakeholders"""
        try:
            # Prepare comprehensive dataset summary
            dataset_summary = self._create_dataset_summary(df, processing_options)
            
            # Prepare analysis insights
            analysis_insights = ""
            if analysis_results:
                analysis_insights = self._create_analysis_insights(analysis_results)
            
            summary_prompt = f"""
            You are writing an executive summary for senior business stakeholders who are not technical.
            
            Dataset Summary:
            {dataset_summary}
            
            Analysis Insights:
            {analysis_insights}
            
            Create a comprehensive executive summary in markdown format with these sections:
            
            # Executive Summary: Data Analysis Report
            
            ## Key Findings
            [3-4 most important insights for business decision makers]
            
            ## Data Quality Assessment
            [Overall data quality and reliability for business decisions]
            
            ## Business Opportunities
            [2-3 specific opportunities identified from the analysis]
            
            ## Risk Factors
            [Any data quality issues or risks to be aware of]
            
            ## Recommendations
            [3-4 specific, actionable recommendations]
            
            ## Technical Summary
            [Brief technical overview for IT/data teams]
            
            Write in clear, business-friendly language. Focus on business value, opportunities, and actionable insights.
            Avoid technical jargon and explain any necessary technical concepts in simple terms.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior business analyst creating executive reports. Write clearly for business stakeholders, focusing on value and actionable insights."},
                    {"role": "user", "content": summary_prompt}
                ]
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"""# Executive Summary: Data Analysis Report

## Error
Unable to generate executive summary due to technical error: {str(e)}

## Manual Summary Required
Please review the analysis results manually and create a custom summary for your stakeholders.

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    def _create_dataset_summary(self, df: pd.DataFrame, processing_options: Optional[Dict] = None) -> str:
        """Create a summary of the dataset characteristics"""
        summary_parts = []
        
        # Basic statistics
        summary_parts.append(f"Dataset contains {len(df):,} records with {len(df.columns)} variables")
        
        # Data types
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        text_count = len(df.select_dtypes(include=['object']).columns)
        date_count = len(df.select_dtypes(include=['datetime64']).columns)
        
        summary_parts.append(f"Data composition: {numeric_count} numeric, {text_count} text, {date_count} date variables")
        
        # Data quality metrics
        missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        duplicate_count = df.duplicated().sum()
        
        summary_parts.append(f"Data quality: {missing_percentage:.1f}% missing values, {duplicate_count} duplicate records")
        
        # Processing applied
        if processing_options:
            applied_processing = [key for key, value in processing_options.items() if value]
            if applied_processing:
                summary_parts.append(f"Processing applied: {', '.join(applied_processing)}")
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        summary_parts.append(f"Dataset size: {memory_mb:.1f} MB")
        
        return '\n'.join(summary_parts)
    
    def _create_analysis_insights(self, analysis_results: Dict) -> str:
        """Create a summary of analysis insights"""
        insights_parts = []
        
        # Statistical insights
        if 'statistical_profile' in analysis_results:
            insights_parts.append("Comprehensive statistical profiling completed")
        
        # Correlation insights
        if 'correlations' in analysis_results and analysis_results['correlations'] is not None:
            corr_summary = self._summarize_correlations(analysis_results['correlations'])
            insights_parts.append(corr_summary)
        
        # Trend insights
        if 'trends' in analysis_results and analysis_results['trends']:
            insights_parts.append(f"Trend analysis: {len(analysis_results['trends'])} significant patterns identified")
        
        # Anomaly insights
        if 'anomalies' in analysis_results and analysis_results['anomalies']:
            insights_parts.append(f"Data quality: {len(analysis_results['anomalies'])} anomalies detected")
        
        # Prediction insights
        if 'predictions' in analysis_results and analysis_results['predictions']:
            good_models = [target for target, model in analysis_results['predictions'].items() 
                          if isinstance(model, dict) and model.get('r2_score', 0) > 0.5]
            if good_models:
                insights_parts.append(f"Predictive models: Good prediction accuracy for {', '.join(good_models[:2])}")
        
        # Clustering insights
        if 'clustering' in analysis_results and 'optimal_clusters' in analysis_results['clustering']:
            cluster_count = analysis_results['clustering']['optimal_clusters']
            insights_parts.append(f"Market segmentation: {cluster_count} natural customer/data segments identified")
        
        # Quality assessment
        if 'quality_assessment' in analysis_results:
            quality = analysis_results['quality_assessment']
            if 'overall_score' in quality:
                score = quality['overall_score']['score']
                grade = quality['overall_score']['grade']
                insights_parts.append(f"Overall data quality: {grade} (score: {score:.2f})")
        
        return '\n'.join(insights_parts) if insights_parts else "Analysis completed successfully"
    
    def generate_business_insights(self, df: pd.DataFrame, question_context: str = "") -> List[str]:
        """Generate business-focused insights from the dataset"""
        try:
            # Prepare dataset context
            dataset_context = self._prepare_dataset_context(df)
            
            insights_prompt = f"""
            Dataset Context:
            {dataset_context}
            
            Context: {question_context}
            
            Generate 3-5 business insights that would be valuable for decision makers.
            Focus on:
            - Revenue opportunities
            - Cost optimization
            - Risk factors
            - Market trends
            - Customer behavior patterns
            - Operational efficiency
            
            Return your response as JSON with this structure:
            {{
                "insights": [
                    "insight 1",
                    "insight 2",
                    "insight 3"
                ]
            }}
            
            Make insights specific, actionable, and business-focused.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a business consultant providing strategic insights from data analysis."},
                    {"role": "user", "content": insights_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            ai_response = json.loads(response.choices[0].message.content)
            return ai_response.get('insights', [])
        
        except Exception as e:
            return [f"Error generating business insights: {str(e)}"]
    
    def suggest_next_analyses(self, df: pd.DataFrame, current_analysis: Optional[Dict] = None) -> List[str]:
        """Suggest next steps for analysis based on current dataset and results"""
        try:
            suggestions = []
            
            # Basic suggestions based on data types
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            text_cols = df.select_dtypes(include=['object']).columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            
            if len(numeric_cols) > 1:
                suggestions.append("Perform regression analysis to identify key drivers")
                suggestions.append("Create predictive models for key business metrics")
            
            if len(text_cols) > 0:
                suggestions.append("Analyze text patterns and categorize unstructured data")
                suggestions.append("Perform sentiment analysis on text fields")
            
            if len(datetime_cols) > 0:
                suggestions.append("Conduct time series analysis for trend forecasting")
                suggestions.append("Analyze seasonal patterns and cyclical trends")
            
            # Suggestions based on current analysis
            if current_analysis:
                if 'correlations' in current_analysis:
                    suggestions.append("Investigate causal relationships behind strong correlations")
                
                if 'anomalies' in current_analysis and current_analysis['anomalies']:
                    suggestions.append("Deep dive into anomalous data points for insights")
                
                if 'clustering' in current_analysis:
                    suggestions.append("Profile customer segments for targeted strategies")
            
            return suggestions[:5]  # Return top 5 suggestions
        
        except Exception:
            return ["Continue exploring the dataset with additional analytical techniques"]
