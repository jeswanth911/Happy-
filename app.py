import streamlit as st
import pandas as pd
import json
import os
from io import StringIO, BytesIO
import traceback

# Import custom modules
from data_processor import DataProcessor
from data_cleaner import DataCleaner
from data_analyzer import DataAnalyzer
from sql_generator import SQLGenerator
from ai_assistant import AIAssistant
from visualization import Visualizer
from utils import format_bytes, validate_file_size

# Configure page
st.set_page_config(
    page_title="Enterprise Data Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'sql_schema' not in st.session_state:
    st.session_state.sql_schema = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def main():
    st.title("🏢 Enterprise Data Analysis Platform")
    st.markdown("Transform your data into actionable business insights with enterprise-grade processing capabilities.")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Module",
        ["Data Upload & Processing", "Data Analysis", "SQL Generation", "AI Assistant", "Export & Reports"]
    )
    
    if page == "Data Upload & Processing":
        data_upload_page()
    elif page == "Data Analysis":
        data_analysis_page()
    elif page == "SQL Generation":
        sql_generation_page()
    elif page == "AI Assistant":
        ai_assistant_page()
    elif page == "Export & Reports":
        export_page()

def data_upload_page():
    st.header("📁 Data Upload & Processing")
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Upload your datasets",
        type=['csv', 'xlsx', 'xls', 'json', 'xml', 'pdf', 'sql'],
        accept_multiple_files=True,
        help="Supported formats: CSV, Excel, JSON, XML, PDF (tables), SQL dumps"
    )
    
    if uploaded_files:
        st.subheader("📋 File Summary")
        
        # Display file information
        for i, file in enumerate(uploaded_files):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{file.name}**")
            with col2:
                st.write(f"{format_bytes(file.size)}")
            with col3:
                st.write(file.type or "Unknown")
        
        # Processing options
        st.subheader("⚙️ Processing Options")
        
        col1, col2 = st.columns(2)
        with col1:
            clean_duplicates = st.checkbox("Remove duplicates", value=True)
            standardize_columns = st.checkbox("Standardize column names", value=True)
            fix_missing_values = st.checkbox("Fix missing values", value=True)
        
        with col2:
            detect_anomalies = st.checkbox("Detect and correct anomalies", value=True)
            mask_pii = st.checkbox("Mask PII data", value=True)
            normalize_schema = st.checkbox("Normalize schema for SQL", value=True)
        
        # Process button
        if st.button("🚀 Process Data", type="primary"):
            try:
                with st.spinner("Processing files... This may take a moment for large datasets."):
                    processor = DataProcessor()
                    cleaner = DataCleaner()
                    
                    # Process each file
                    processed_datasets = []
                    for file in uploaded_files:
                        try:
                            # Reset file pointer
                            file.seek(0)
                            
                            # Process file based on type
                            df = processor.process_file(file)
                            
                            if df is not None and not df.empty:
                                # Clean data based on options
                                if clean_duplicates:
                                    df = cleaner.remove_duplicates(df)
                                
                                if standardize_columns:
                                    df = cleaner.standardize_columns(df)
                                
                                if fix_missing_values:
                                    df = cleaner.fix_missing_values(df)
                                
                                if detect_anomalies:
                                    df = cleaner.detect_and_correct_anomalies(df)
                                
                                if mask_pii:
                                    df = cleaner.mask_pii(df)
                                
                                processed_datasets.append({
                                    'name': file.name,
                                    'data': df,
                                    'original_shape': df.shape
                                })
                        
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
                            continue
                    
                    if processed_datasets:
                        # Combine datasets if multiple files
                        if len(processed_datasets) == 1:
                            final_df = processed_datasets[0]['data']
                            dataset_name = processed_datasets[0]['name']
                        else:
                            # Attempt to combine datasets
                            try:
                                final_df = cleaner.combine_datasets([d['data'] for d in processed_datasets])
                                dataset_name = "Combined Dataset"
                            except Exception as e:
                                st.error(f"Could not combine datasets: {str(e)}")
                                final_df = processed_datasets[0]['data']
                                dataset_name = processed_datasets[0]['name']
                        
                        # Store in session state
                        st.session_state.processed_data = {
                            'dataframe': final_df,
                            'name': dataset_name,
                            'processing_options': {
                                'clean_duplicates': clean_duplicates,
                                'standardize_columns': standardize_columns,
                                'fix_missing_values': fix_missing_values,
                                'detect_anomalies': detect_anomalies,
                                'mask_pii': mask_pii,
                                'normalize_schema': normalize_schema
                            }
                        }
                        
                        st.success(f"✅ Successfully processed {len(processed_datasets)} file(s)")
                        
                        # Display processing summary
                        st.subheader("📊 Processing Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Rows", f"{len(final_df):,}")
                        with col2:
                            st.metric("Columns", len(final_df.columns))
                        with col3:
                            st.metric("Memory Usage", format_bytes(final_df.memory_usage(deep=True).sum()))
                        with col4:
                            st.metric("Data Types", final_df.dtypes.nunique())
                        
                        # Preview data
                        st.subheader("👀 Data Preview")
                        st.dataframe(final_df.head(100), use_container_width=True)
                        
                        # Data quality metrics
                        st.subheader("🎯 Data Quality Metrics")
                        quality_metrics = cleaner.get_quality_metrics(final_df)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Completeness", f"{quality_metrics['completeness']:.1%}")
                        with col2:
                            st.metric("Uniqueness", f"{quality_metrics['uniqueness']:.1%}")
                        with col3:
                            st.metric("Validity", f"{quality_metrics['validity']:.1%}")
                    
                    else:
                        st.error("No valid data could be processed from the uploaded files.")
            
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.error("Please check your file formats and try again.")

def data_analysis_page():
    st.header("📈 Data Analysis")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please upload and process data first.")
        return
    
    df = st.session_state.processed_data['dataframe']
    
    if st.button("🔍 Run Comprehensive Analysis", type="primary"):
        try:
            with st.spinner("Performing comprehensive analysis..."):
                analyzer = DataAnalyzer()
                visualizer = Visualizer()
                
                # Perform analysis
                analysis_results = analyzer.comprehensive_analysis(df)
                st.session_state.analysis_results = analysis_results
                
                # Statistical Profile
                st.subheader("📊 Statistical Profile")
                st.json(analysis_results['statistical_profile'])
                
                # Correlations
                if 'correlations' in analysis_results:
                    st.subheader("🔗 Correlation Analysis")
                    corr_fig = visualizer.create_correlation_heatmap(analysis_results['correlations'])
                    st.plotly_chart(corr_fig, use_container_width=True)
                
                # Anomalies
                if analysis_results['anomalies']:
                    st.subheader("⚠️ Detected Anomalies")
                    for anomaly in analysis_results['anomalies']:
                        st.warning(anomaly)
                
                # Trends
                if analysis_results['trends']:
                    st.subheader("📈 Trend Analysis")
                    for trend in analysis_results['trends']:
                        st.info(trend)
                
                # Business Insights
                st.subheader("💡 Business Insights")
                insights = analyzer.generate_business_insights(df, analysis_results)
                for insight in insights:
                    st.success(insight)
                
                # Predictions
                if 'predictions' in analysis_results:
                    st.subheader("🔮 Predictive Models")
                    st.json(analysis_results['predictions'])
        
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
    
    # Display existing analysis results
    if st.session_state.analysis_results:
        st.subheader("📋 Current Analysis Results")
        
        # Create tabs for different analysis sections
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Visualizations", "Insights", "Export"])
        
        with tab1:
            results = st.session_state.analysis_results
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Numeric Columns", len([col for col in df.columns if df[col].dtype in ['int64', 'float64']]))
            with col2:
                st.metric("Categorical Columns", len([col for col in df.columns if df[col].dtype == 'object']))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Duplicate Rows", df.duplicated().sum())
        
        with tab2:
            visualizer = Visualizer()
            
            # Distribution plots for numeric columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 0:
                st.subheader("Distribution Analysis")
                selected_col = st.selectbox("Select column for distribution", numeric_cols)
                if selected_col:
                    dist_fig = visualizer.create_distribution_plot(df[selected_col], selected_col)
                    st.plotly_chart(dist_fig, use_container_width=True)
        
        with tab3:
            if 'business_insights' in st.session_state.analysis_results:
                for insight in st.session_state.analysis_results['business_insights']:
                    st.info(insight)
        
        with tab4:
            st.download_button(
                "Download Analysis Report (JSON)",
                data=json.dumps(st.session_state.analysis_results, indent=2, default=str),
                file_name="analysis_report.json",
                mime="application/json"
            )

def sql_generation_page():
    st.header("🗄️ SQL Generation")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please upload and process data first.")
        return
    
    df = st.session_state.processed_data['dataframe']
    dataset_name = st.session_state.processed_data['name']
    
    # SQL generation options
    col1, col2 = st.columns(2)
    with col1:
        table_name = st.text_input("Table Name", value=dataset_name.split('.')[0].lower().replace(' ', '_'))
    with col2:
        include_indexes = st.checkbox("Include indexes", value=True)
    
    if st.button("🔨 Generate SQL Schema", type="primary"):
        try:
            with st.spinner("Generating SQL schema..."):
                sql_generator = SQLGenerator()
                
                # Generate schema
                table_name_safe = table_name if table_name else "data_table"
                schema_result = sql_generator.generate_schema(df, table_name_safe, include_indexes)
                st.session_state.sql_schema = schema_result
                
                # Display CREATE TABLE statement
                st.subheader("🏗️ CREATE TABLE Statement")
                st.code(schema_result['create_table'], language='sql')
                
                # Display sample INSERT statements
                st.subheader("📝 Sample INSERT Statements")
                st.code(schema_result['sample_inserts'], language='sql')
                
                # Schema summary
                st.subheader("📋 Schema Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Columns", len(schema_result['column_mapping']))
                with col2:
                    st.metric("Primary Keys", len(schema_result.get('primary_keys', [])))
                with col3:
                    st.metric("Indexes", len(schema_result.get('indexes', [])))
                
                # Column mapping details
                st.subheader("🗂️ Column Mapping")
                mapping_df = pd.DataFrame([
                    {
                        'Original Column': col,
                        'SQL Column': details['sql_name'],
                        'Data Type': details['sql_type'],
                        'Nullable': details['nullable'],
                        'Comments': details.get('comments', '')
                    }
                    for col, details in schema_result['column_mapping'].items()
                ])
                st.dataframe(mapping_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"SQL generation failed: {str(e)}")
    
    # Export options
    if st.session_state.sql_schema:
        st.subheader("💾 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "Download CREATE TABLE",
                data=st.session_state.sql_schema['create_table'],
                file_name=f"{table_name}_schema.sql",
                mime="text/plain"
            )
        
        with col2:
            st.download_button(
                "Download INSERT Statements",
                data=st.session_state.sql_schema['full_inserts'],
                file_name=f"{table_name}_data.sql",
                mime="text/plain"
            )
        
        with col3:
            # Generate full SQL dump
            full_sql = f"{st.session_state.sql_schema['create_table']}\n\n{st.session_state.sql_schema['full_inserts']}"
            st.download_button(
                "Download Complete SQL",
                data=full_sql,
                file_name=f"{table_name}_complete.sql",
                mime="text/plain"
            )

def ai_assistant_page():
    st.header("🤖 AI Assistant")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please upload and process data first.")
        return
    
    df = st.session_state.processed_data['dataframe']
    
    # Initialize AI Assistant
    try:
        ai_assistant = AIAssistant()
        
        # Dataset context
        st.subheader("📊 Dataset Context")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Numeric Cols", len(df.select_dtypes(include=['number']).columns))
        with col4:
            st.metric("Text Cols", len(df.select_dtypes(include=['object']).columns))
        
        # Chat interface
        st.subheader("💬 Ask Questions About Your Data")
        
        # Example questions
        with st.expander("💡 Example Questions"):
            st.write("• What are the key trends in this dataset?")
            st.write("• Which columns have the strongest correlations?")
            st.write("• What outliers can you identify?")
            st.write("• Generate a summary for executives")
            st.write("• What predictions can you make?")
            st.write("• Show me the top 10 records by [column name]")
        
        # Chat input
        user_question = st.text_input(
            "Your Question:",
            placeholder="Ask anything about your dataset...",
            key="user_question"
        )
        
        if st.button("Ask AI", type="primary") and user_question:
            try:
                with st.spinner("AI is analyzing your question..."):
                    # Get AI response
                    response = ai_assistant.answer_question(df, user_question, st.session_state.analysis_results)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'response': response,
                        'timestamp': pd.Timestamp.now()
                    })
                    
                    # Display response
                    st.subheader("🎯 AI Response")
                    
                    if 'summary' in response:
                        st.info(response['summary'])
                    
                    if 'sql_query' in response:
                        st.subheader("🔍 Generated SQL Query")
                        st.code(response['sql_query'], language='sql')
                    
                    if 'results' in response:
                        st.subheader("📊 Query Results")
                        if isinstance(response['results'], pd.DataFrame):
                            st.dataframe(response['results'], use_container_width=True)
                        else:
                            st.write(response['results'])
                    
                    if 'chart' in response:
                        st.subheader("📈 Visualization")
                        st.plotly_chart(response['chart'], use_container_width=True)
                    
                    if 'recommendations' in response:
                        st.subheader("💡 Recommendations")
                        for rec in response['recommendations']:
                            st.success(rec)
            
            except Exception as e:
                st.error(f"AI Assistant error: {str(e)}")
        
        # Chat history
        if st.session_state.chat_history:
            st.subheader("📝 Chat History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                with st.expander(f"Q: {chat['question'][:50]}..." if len(chat['question']) > 50 else f"Q: {chat['question']}"):
                    st.write(f"**Asked:** {chat['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Question:** {chat['question']}")
                    
                    if isinstance(chat['response'], dict):
                        if 'summary' in chat['response']:
                            st.write(f"**Answer:** {chat['response']['summary']}")
                        if 'sql_query' in chat['response']:
                            st.code(chat['response']['sql_query'], language='sql')
                    else:
                        st.write(f"**Answer:** {chat['response']}")
    
    except Exception as e:
        st.error(f"Could not initialize AI Assistant: {str(e)}")
        st.info("Make sure your OpenAI API key is configured in the environment variables.")

def export_page():
    st.header("📤 Export & Reports")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please upload and process data first.")
        return
    
    df = st.session_state.processed_data['dataframe']
    dataset_name = st.session_state.processed_data['name']
    
    st.subheader("📋 Executive Summary")
    
    if st.button("📊 Generate Executive Summary", type="primary"):
        try:
            with st.spinner("Generating executive summary..."):
                ai_assistant = AIAssistant()
                
                # Generate comprehensive summary
                summary = ai_assistant.generate_executive_summary(
                    df, 
                    st.session_state.analysis_results,
                    st.session_state.processed_data['processing_options']
                )
                
                st.subheader("📈 Executive Summary Report")
                st.markdown(summary)
                
                # Download options
                st.download_button(
                    "Download Executive Summary",
                    data=summary,
                    file_name=f"{dataset_name}_executive_summary.md",
                    mime="text/markdown"
                )
        
        except Exception as e:
            st.error(f"Failed to generate summary: {str(e)}")
    
    # Data export options
    st.subheader("💾 Data Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV export
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📄 Download CSV",
            data=csv_data,
            file_name=f"{dataset_name}_cleaned.csv",
            mime="text/csv"
        )
    
    with col2:
        # Excel export
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            
            # Add summary sheet if analysis exists
            if st.session_state.analysis_results:
                summary_df = pd.DataFrame([
                    ['Dataset Name', dataset_name],
                    ['Total Rows', len(df)],
                    ['Total Columns', len(df.columns)],
                    ['Memory Usage', format_bytes(df.memory_usage(deep=True).sum())],
                    ['Processing Date', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')]
                ])
                summary_df.to_excel(writer, sheet_name='Summary', index=False, header=False)
        
        st.download_button(
            "📊 Download Excel",
            data=excel_buffer.getvalue(),
            file_name=f"{dataset_name}_cleaned.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col3:
        # JSON export
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            "🔧 Download JSON",
            data=json_data,
            file_name=f"{dataset_name}_cleaned.json",
            mime="application/json"
        )
    
    # Complete report package
    st.subheader("📦 Complete Analysis Package")
    
    if st.button("📋 Generate Complete Report Package"):
        try:
            # Create comprehensive report
            report_data = {
                'dataset_info': {
                    'name': dataset_name,
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.to_dict(),
                    'processing_date': pd.Timestamp.now().isoformat()
                },
                'processing_options': st.session_state.processed_data['processing_options'],
                'analysis_results': st.session_state.analysis_results,
                'sql_schema': st.session_state.sql_schema,
                'chat_history': st.session_state.chat_history
            }
            
            st.download_button(
                "📦 Download Complete Report",
                data=json.dumps(report_data, indent=2, default=str),
                file_name=f"{dataset_name}_complete_analysis.json",
                mime="application/json"
            )
            
            st.success("✅ Complete analysis package ready for download!")
        
        except Exception as e:
            st.error(f"Failed to generate complete report: {str(e)}")

if __name__ == "__main__":
    main()
