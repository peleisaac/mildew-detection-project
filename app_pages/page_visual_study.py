import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json


def page_visual_study_body():
    """
    Display visual differentiation study for cherry leaf classification
    Business Requirement 1: Understanding visual differences between healthy and mildew-infected leaves
    """

    st.write("### Visual Differentiation Study 🔬")

    st.info(
        "**Business Requirement 1**: The client is interested in conducting a study to "
        "visually differentiate a cherry leaf that is healthy from one that contains powdery mildew."
    )

    # Project hypothesis section
    with st.expander("📊 Project Hypothesis", expanded=False):
        st.markdown(
            """
        **Hypothesis 1**: Cherry leaves infected with powdery mildew display distinct visual characteristics 
        that can be differentiated from healthy leaves through image analysis.
        
        **Expected Visual Differences**:
        - Infected leaves will show white, powdery patches on the surface
        - Healthy leaves will maintain uniform green coloration
        - Texture differences will be observable between classes
        
        **Validation Method**: Average image analysis and visual comparison studies will reveal 
        distinguishable patterns between healthy and infected leaves.
        """
        )

    # Average Images Analysis Section
    st.write("---")
    st.write("## 📸 Average Images Analysis")
    st.markdown(
        """
    The average images below represent the typical appearance of each class. These are computed by 
    averaging pixel values across all images in each category, revealing the most common visual patterns.
    """
    )

    # Create three columns for average images
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("#### 🌱 Average Healthy Leaf")
        try:
            avg_healthy_path = "outputs/plots/avg_healthy.png"
            if os.path.exists(avg_healthy_path):
                avg_healthy = Image.open(avg_healthy_path)
                st.image(avg_healthy)
                st.success(
                    """
                **Key Observations:**
                - Uniform green coloration
                - Clear leaf structure
                - Smooth surface appearance
                - Well-defined veining patterns
                """
                )
            else:
                st.warning(
                    "Average healthy leaf image not found. Please run data visualization notebook."
                )
        except Exception as e:
            st.error(f"Error loading average healthy image: {str(e)}")

    with col2:
        st.markdown("#### 🦠 Average Mildew-Infected Leaf")
        try:
            avg_mildew_path = "outputs/plots/avg_mildew.png"
            if os.path.exists(avg_mildew_path):
                avg_mildew = Image.open(avg_mildew_path)
                st.image(avg_mildew)
                st.error(
                    """
                **Key Observations:**
                - White powdery patches visible
                - Irregular surface texture
                - Discoloration patterns
                - Fuzzy, dusty appearance
                """
                )
            else:
                st.warning(
                    "Average mildew leaf image not found. Please run data visualization notebook."
                )
        except Exception as e:
            st.error(f"Error loading average mildew image: {str(e)}")

    with col3:
        st.markdown("#### 🔍 Visual Difference Map")
        try:
            diff_path = "outputs/plots/abs_diff.png"
            if os.path.exists(diff_path):
                diff_image = Image.open(diff_path)
                st.image(diff_image)
                st.info(
                    """
                **Interpretation:**
                - Bright areas show maximum differences
                - Highlights infection signature regions
                - Reveals diagnostic patterns
                - Guides feature importance understanding
                """
                )
            else:
                st.warning(
                    "Difference visualization not found. Please run data visualization notebook."
                )
        except Exception as e:
            st.error(f"Error loading difference image: {str(e)}")

    # Individual Sample Comparison
    st.write("---")
    st.write("## 🔬 Individual Sample Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Representative Healthy Sample")
        try:
            healthy_sample_path = "outputs/plots/healthy.png"
            if os.path.exists(healthy_sample_path):
                healthy_sample = Image.open(healthy_sample_path)
                st.image(healthy_sample)
                st.markdown(
                    """
                **Healthy Leaf Characteristics:**
                - ✅ Vibrant green coloration
                - ✅ Smooth, even surface texture
                - ✅ Clear leaf boundaries
                - ✅ No visible spots or patches
                - ✅ Natural leaf structure integrity
                """
                )
            else:
                st.warning("Individual healthy sample not available.")
        except Exception as e:
            st.error(f"Error loading healthy sample: {str(e)}")

    with col2:
        st.markdown("#### Representative Mildew Sample")
        try:
            mildew_sample_path = "outputs/plots/mildew.png"
            if os.path.exists(mildew_sample_path):
                mildew_sample = Image.open(mildew_sample_path)
                st.image(mildew_sample)
                st.markdown(
                    """
                **Mildew Infection Indicators:**
                - ❌ White powdery coating present
                - ❌ Surface texture irregularities
                - ❌ Discoloration and patches
                - ❌ Fuzzy, dusty appearance
                - ❌ Compromised leaf structure
                """
                )
            else:
                st.warning("Individual mildew sample not available.")
        except Exception as e:
            st.error(f"Error loading mildew sample: {str(e)}")

    # Sample Montages Section
    st.write("---")
    st.write("## 🖼️ Sample Collections")
    st.markdown(
        """
    Below are montages showing multiple examples from each class, demonstrating the consistency 
    of visual patterns within each category and the clear distinctions between them.
    """
    )

    # Create tabs for sample montages
    tab1, tab2 = st.tabs(["🌱 Healthy Samples", "🦠 Mildew Samples"])

    with tab1:
        st.markdown("### Healthy Cherry Leaf Collection")
        try:
            healthy_grid_path = "outputs/plots/healthy_samples_grid.png"
            if os.path.exists(healthy_grid_path):
                healthy_grid = Image.open(healthy_grid_path)
                st.image(
                    healthy_grid,
                    caption="Montage of healthy cherry leaf samples showing consistent green coloration and clear structure",
                )
                st.success(
                    """
                **Analysis of Healthy Samples:**
                - Consistent green coloration across all samples
                - Uniform leaf structure and texture
                - No presence of white patches or abnormal spots
                - Natural variation in leaf size and orientation while maintaining health indicators
                """
                )
            else:
                st.warning(
                    "Healthy samples montage not found. Please generate using data visualization notebook."
                )
        except Exception as e:
            st.error(f"Error loading healthy samples grid: {str(e)}")

    with tab2:
        st.markdown("### Mildew-Infected Cherry Leaf Collection")
        try:
            mildew_grid_path = "outputs/plots/powdery_mildew_samples_grid.png"
            if os.path.exists(mildew_grid_path):
                mildew_grid = Image.open(mildew_grid_path)
                st.image(
                    mildew_grid,
                    caption="Montage of mildew-infected cherry leaf samples showing characteristic white powdery patches",
                )
                st.error(
                    """
                **Analysis of Infected Samples:**
                - Visible white, powdery mildew patches across samples
                - Varying degrees of infection severity
                - Consistent presence of surface texture changes
                - Clear deviation from healthy leaf appearance patterns
                """
                )
            else:
                st.warning(
                    "Mildew samples montage not found. Please generate using data visualization notebook."
                )
        except Exception as e:
            st.error(f"Error loading mildew samples grid: {str(e)}")

    # Dataset Statistics and Distribution
    st.write("---")
    st.write("## 📊 Dataset Overview")

    # Create interactive dataset statistics
    try:
        # Load dataset statistics if available
        stats_path = "outputs/dataset_stats.json"
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)

            col1, col2 = st.columns(2)

            with col1:
                # Class distribution pie chart
                fig_pie = px.pie(
                    values=[
                        stats.get("healthy_leaves", 0),
                        stats.get("powdery_mildew_leaves", 0),
                    ],
                    names=["Healthy", "Powdery Mildew"],
                    title="Dataset Class Distribution",
                    color_discrete_map={
                        "Healthy": "#2E8B57",
                        "Powdery Mildew": "#DC143C",
                    },
                )
                st.plotly_chart(fig_pie)

            with col2:
                # Display key metrics
                st.metric("Total Images", stats.get("total_images", "N/A"))
                st.metric("Healthy Images", stats.get("healthy_leaves", "N/A"))
                st.metric(
                    "Powdery Mildew Images", stats.get("powdery_mildew_leaves", "N/A")
                )
                st.metric(
                    "Image Dimensions",
                    f"{stats.get('image_height', '256')}x{stats.get('image_width', '256')}",
                )

            st.write("#### 📊 Dataset Image count per class and Subset")
            st.image("outputs/plots/class_counts.png")

        else:
            st.info(
                "Dataset statistics not available. Run data collection notebook to generate statistics."
            )
    except Exception as e:
        st.warning(f"Could not load dataset statistics: {str(e)}")


    # Scatter plot of image dimension
    st.markdown("#### 📏 Scatterplot of Image Dimensions")
    st.image("outputs/plots/image_dimension_scatter.png", )

    
    # Side-by-side Comparison Section
    st.write("---")
    st.write("## ⚖️ Direct Visual Comparison")

    try:
        comparison_path = "outputs/plots/healthy_vs_mildew.png"
        if os.path.exists(comparison_path):
            comparison_image = Image.open(comparison_path)
            st.image(
                comparison_image,
                caption="Direct comparison highlighting key visual differences between healthy and mildew-infected leaves",
            )
        else:
            st.warning("Comparison image not available.")
    except Exception as e:
        st.error(f"Error loading comparison image: {str(e)}")

    # Key Findings and Conclusions
    st.write("---")
    st.write("## 🎯 Key Findings & Conclusions")

    col1, col2 = st.columns(2)

    with col1:
        st.success(
            """
        ### ✅ Hypothesis Validation Results
        
        **Confirmed Visual Differences:**
        - Clear distinction in surface texture patterns
        - Consistent color variation between classes
        - Identifiable mildew signature patterns
        - Reliable visual markers for classification
        
        **Statistical Validation:**
        - Average images show distinct class characteristics
        - Difference mapping reveals diagnostic regions
        - Sample consistency within each class confirmed
        """
        )

    with col2:
        st.info(
            """
        ### 🔍 Business Impact
        
        **Diagnostic Capabilities:**
        - Visual patterns enable automated detection
        - Consistent markers across sample population
        - Reliable basis for ML model training
        - Scalable detection methodology
        
        **Practical Applications:**
        - Early infection detection possible
        - Reduced manual inspection time
        - Consistent diagnostic criteria
        - Agricultural decision support
        """
        )

    # Technical Notes
    with st.expander("🛠️ Technical Implementation Notes", expanded=False):
        st.markdown(
            """
        **Image Processing Pipeline:**
        - Images standardized to consistent dimensions
        - Pixel value averaging for class representatives
        - Absolute difference computation for variation mapping
        - Statistical analysis of visual feature distributions
        
        **Visualization Techniques:**
        - Average image computation across class samples
        - Pixel-wise difference analysis
        - Montage creation for pattern recognition
        - Interactive statistical dashboards
        
        **Quality Assurance:**
        - Manual verification of sample classifications
        - Consistency checks across image batches
        - Validation of visual pattern hypotheses
        - Cross-reference with domain expertise
        """
        )
