import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
from datetime import datetime
from src.machine_learning.predictive_analysis import (
    resize_input_image,
    load_model_and_predict,
)


def page_batch_analysis():
    """Batch analysis page for multiple images"""

    st.write("## 📁 Batch Analysis")

    st.info(
        """
    Upload multiple cherry leaf images for batch processing. Get comprehensive analysis 
    reports, statistics, and downloadable results for your entire dataset.
    """
    )

    # File uploader for multiple files
    st.markdown("### 📄 Upload Multiple Images")
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Select multiple cherry leaf images for batch analysis",
    )

    if uploaded_files:
        st.success(f"📊 {len(uploaded_files)} images uploaded successfully!")

        # Analysis settings
        st.markdown("### ⚙️ Analysis Settings")

        col1, col2, col3 = st.columns(3)

        with col1:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Minimum confidence level for predictions",
            )

        with col2:
            batch_size = st.selectbox(
                "Processing Batch Size",
                options=[10, 25, 50, 100],
                index=1,
                help="Number of images to process at once",
            )

        with col3:
            save_results = st.checkbox(
                "Save Individual Results",
                value=True,
                help="Save detailed results for each image",
            )

        # Start batch analysis
        if st.button("🚀 Start Batch Analysis", type="primary"):

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Results storage
            results = []

            # Process each image
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(
                    f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})"
                )

                try:
                    # Load and process image
                    img = Image.open(uploaded_file).convert("RGB")
                    image_array = resize_input_image(img)

                    # Actual prediction
                    prediction, confidence, prediction_proba = load_model_and_predict(
                        image_array
                    )
                    prediction_label = (
                        "Infected" if prediction == "powdery_mildew" else "Healthy"
                    )

                    # Store results
                    result = {
                        "filename": uploaded_file.name,
                        "prediction": prediction_label,
                        "confidence": round(confidence, 2),
                        "status": (
                            "High Confidence"
                            if confidence >= confidence_threshold * 100
                            else "Low Confidence"
                        ),
                        "size": f"{img.size[0]}x{img.size[1]}",
                        "format": img.format,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    results.append(result)

                except Exception as e:
                    st.warning(f"Could not process {uploaded_file.name}: {str(e)}")

                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.text("✅ Batch analysis completed!")

            # Convert results to DataFrame
            df_results = pd.DataFrame(results)

            # Display results
            st.markdown("---")
            st.markdown("### 📊 Batch Analysis Results")

            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_images = len(df_results)
                st.metric("Total Images", total_images)

            with col2:
                healthy_count = len(df_results[df_results["prediction"] == "Healthy"])
                st.metric(
                    "Healthy Leaves",
                    healthy_count,
                    f"{healthy_count/total_images*100:.1f}%",
                )

            with col3:
                infected_count = len(df_results[df_results["prediction"] == "Infected"])
                st.metric(
                    "Infected Leaves",
                    infected_count,
                    f"{infected_count/total_images*100:.1f}%",
                )

            with col4:
                high_conf_count = len(
                    df_results[df_results["status"] == "High Confidence"]
                )
                st.metric(
                    "High Confidence",
                    high_conf_count,
                    f"{high_conf_count/total_images*100:.1f}%",
                )

            # Visualizations
            st.markdown("#### 📈 Analysis Visualizations")

            tab1, tab2, tab3 = st.tabs(
                ["Distribution", "Confidence Levels", "Timeline"]
            )

            with tab1:
                # Prediction distribution pie chart
                fig_pie = px.pie(
                    df_results,
                    names="prediction",
                    title="Prediction Distribution",
                    color_discrete_map={"Healthy": "green", "Infected": "red"},
                )
                st.plotly_chart(fig_pie)
                st.markdown("""
**📊 Distribution Chart Analysis:**
This pie chart visualizes the proportion of healthy vs infected leaves in your batch. 
Green represents healthy leaves, red represents infected leaves. The percentages help 
identify the infection rate across your sample, with higher red percentages indicating 
more widespread infection requiring immediate attention.
""")

            with tab2:
                # Confidence distribution histogram
                fig_hist = px.histogram(
                    df_results,
                    x="confidence",
                    color="prediction",
                    title="Confidence Score Distribution",
                    nbins=20,
                    color_discrete_map={"Healthy": "green", "Infected": "red"},
                )
                fig_hist.add_vline(
                    x=confidence_threshold * 100,
                    line_dash="dash",
                    annotation_text=f"Threshold: {confidence_threshold*100}%",
                )
                st.plotly_chart(fig_hist)
                st.markdown("""
**📊 Confidence Histogram Analysis:**
This histogram shows the distribution of prediction confidence scores across all images. 
The vertical dashed line represents your confidence threshold. Bars to the right indicate 
high-confidence predictions suitable for automated action, while bars to the left may 
require manual verification. Color coding helps identify confidence levels by prediction type.
""")

            with tab3:
                # Processing timeline (if timestamps are available)
                fig_timeline = px.scatter(
                    df_results,
                    x="timestamp",
                    y="confidence",
                    color="prediction",
                    title="Processing Timeline",
                    color_discrete_map={"Healthy": "green", "Infected": "red"},
                )
                st.plotly_chart(fig_timeline)
                st.markdown("""
**📊 Timeline Scatter Plot Analysis:**
This scatter plot tracks prediction confidence over processing time, helping identify 
any patterns or quality issues during batch processing. Each point represents one image, 
with Y-axis showing confidence and color indicating health status. Consistent confidence 
levels across time indicate stable model performance.
""")
                

            # Detailed results table
            st.markdown("#### 📋 Detailed Results")

            # Filter options
            col1, col2 = st.columns(2)

            with col1:
                filter_prediction = st.multiselect(
                    "Filter by Prediction",
                    options=["Healthy", "Infected"],
                    default=["Healthy", "Infected"],
                )

            with col2:
                filter_status = st.multiselect(
                    "Filter by Status",
                    options=["High Confidence", "Low Confidence"],
                    default=["High Confidence", "Low Confidence"],
                )

            # Apply filters
            filtered_df = df_results[
                (df_results["prediction"].isin(filter_prediction))
                & (df_results["status"].isin(filter_status))
            ]

            # Display filtered table
            st.dataframe(filtered_df, hide_index=True)
            st.markdown("""
**📋 Results Table Interpretation:**
This comprehensive table provides detailed analysis results for each processed image. 
Key columns include prediction confidence, processing status, and image metadata. 
Use the filters above to focus on specific prediction types or confidence levels. 
High confidence results (>80%) are suitable for immediate action, while lower 
confidence predictions may warrant manual review.
""")
            

            # Action items for infected leaves
            infected_leaves = df_results[df_results["prediction"] == "Infected"]
            if len(infected_leaves) > 0:
                st.markdown("#### ⚠️ Action Required - Infected Leaves Detected")

                st.error(
                    f"""
                **{len(infected_leaves)} infected leaves detected!**
                
                Immediate actions recommended:
                - Review high-confidence infected predictions first
                - Isolate affected plants to prevent spread
                - Apply appropriate fungicide treatment
                - Monitor surrounding plants closely
                """
                )

                # Priority list for infected leaves
                priority_infected = infected_leaves.sort_values(
                    "confidence", ascending=False
                )

                with st.expander("🚨 Priority Treatment List"):
                    for idx, row in priority_infected.iterrows():
                        priority_level = (
                            "🔴 HIGH" if row["confidence"] > 80 else "🟡 MEDIUM"
                        )
                        st.write(
                            f"{priority_level} - {row['filename']} (Confidence: {row['confidence']:.1f}%)"
                        )
                        

            # Download options
            st.markdown("---")
            st.markdown("#### 💾 Download Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                # CSV download
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV Report",
                    data=csv,
                    file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

            with col2:
                # JSON download
                json_data = df_results.to_json(orient="records", indent=2)
                st.download_button(
                    label="📋 Download JSON Report",
                    data=json_data,
                    file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )

            with col3:
                # Summary report
                summary_report = f"""
BATCH ANALYSIS SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW:
- Total Images Processed: {total_images}
- Healthy Leaves: {healthy_count} ({healthy_count/total_images*100:.1f}%)
- Infected Leaves: {infected_count} ({infected_count/total_images*100:.1f}%)
- High Confidence Predictions: {high_conf_count} ({high_conf_count/total_images*100:.1f}%)

RECOMMENDATIONS:
{'- IMMEDIATE ACTION REQUIRED: Treat infected leaves with fungicide' if infected_count > 0 else '- Continue regular monitoring'}
{'- Isolate affected plants to prevent spread' if infected_count > 0 else '- Maintain current plant care practices'}
- Monitor all plants regularly for early detection
- Ensure proper plant spacing and air circulation
                """

                st.download_button(
                    label="📄 Download Summary",
                    data=summary_report,
                    file_name=f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                )

    else:
        # Instructions when no files uploaded
        st.markdown("---")
        st.markdown("### 📋 How to Use Batch Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            **Step-by-Step Guide:**
            1. 📤 Select multiple cherry leaf images
            2. ⚙️ Configure analysis settings
            3. 🚀 Start batch processing
            4. 📊 Review comprehensive results
            5. 💾 Download reports and data
            6. 🚨 Take action on infected leaves
            """
            )

        with col2:
            st.markdown(
                """
            **Benefits of Batch Analysis:**
            - ⚡ Process multiple images quickly
            - 📈 Get statistical overviews
            - 🎯 Identify priority cases
            - 📊 Export data for records
            - 🔍 Filter and sort results
            - 📋 Generate summary reports
            """
            )

        # Sample batch data
        st.markdown("---")
        st.markdown("### 🎯 Sample Batch Results")
        st.info("Here's what your batch analysis results would look like:")

        # Create sample data
        sample_data = {
            "filename": [
                "leaf_001.jpg",
                "leaf_002.jpg",
                "leaf_003.jpg",
                "leaf_004.jpg",
                "leaf_005.jpg",
            ],
            "prediction": ["Healthy", "Infected", "Healthy", "Infected", "Healthy"],
            "confidence": [94.2, 87.5, 91.8, 92.1, 88.9],
            "status": [
                "High Confidence",
                "High Confidence",
                "High Confidence",
                "High Confidence",
                "High Confidence",
            ],
        }
        sample_df = pd.DataFrame(sample_data)

        st.dataframe(sample_df, hide_index=True)

        # Tips section
        st.markdown("---")
        st.markdown("### 💡 Pro Tips")

        st.success(
            """
        **Maximize Batch Analysis Efficiency:**
        - Upload images in smaller batches (25-50) for faster processing
        - Ensure consistent image quality for better results
        - Use descriptive filenames for easier identification
        - Set appropriate confidence thresholds based on your needs
        - Save results regularly for large datasets
        """
        )
