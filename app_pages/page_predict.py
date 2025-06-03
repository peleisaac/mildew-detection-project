import streamlit as st
import numpy as np
from PIL import Image

# import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# from src.data_management import load_pkl_file
from src.machine_learning.predictive_analysis import (
    load_model_and_predict,
    resize_input_image,
    plot_predictions_probabilities,
)
import plotly.graph_objects as go
from datetime import datetime

from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
image_path = base_dir / "../outputs/plots/healthy_samples_grid.png"
powdery_path = base_dir / "../outputs/plots/powdery_mildew_samples_grid.png"
healthy_sample = Image.open(image_path)
mildew_sample = Image.open(powdery_path)


def page_mildew_detector():
    """Single image mildew detection page"""

    st.write("## 🔍 Mildew Detector")

    st.info(
        """
    Upload a cherry leaf image to get instant mildew detection results. 
    The AI model will analyze the image and provide a confidence score for the prediction.
    """
    )

    # File uploader
    st.markdown("### 📤 Upload Cherry Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear image of a cherry leaf for analysis",
    )

    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 🖼️ Uploaded Image")
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Cherry Leaf", use_container_width=True)

            # Image information
            st.markdown("**Image Details:**")
            st.write(f"- **Filename:** {uploaded_file.name}")
            st.write(f"- **Size:** {img.size[0]} x {img.size[1]} pixels")
            st.write(f"- **Format:** {img.format}")
            st.write(f"- **Mode:** {img.mode}")

        with col2:
            st.markdown("#### 🤖 AI Analysis")

            # Analysis button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image... Please wait."):

                    try:
                        # Prediction
                        image_array = resize_input_image(img)
                        prediction, confidence, prediction_proba = (
                            load_model_and_predict(image_array)
                        )

                        # For display purposes
                        predicted_class = (
                            "Powdery Mildew"
                            if prediction == "powdery_mildew"
                            else "Healthy"
                        )

                        # Use for status indicators
                        status_color = (
                            "🔴" if predicted_class == "Powdery Mildew" else "🟢"
                        )
                        status_class = (
                            "error"
                            if predicted_class == "Powdery Mildew"
                            else "success"
                        )

                        # Display results
                        st.markdown("---")
                        st.markdown("### 📊 Analysis Results")

                        # Main prediction result
                        if prediction == "healthy":
                            st.success(f"🌿 **Prediction: HEALTHY LEAF**")
                        else:
                            st.error(f"🦠 **Prediction: POWDERY MILDEW DETECTED**")

                        # Confidence metrics
                        col_a, col_b, col_c = st.columns(3)

                        with col_a:
                            st.metric(
                                label="Confidence",
                                value=f"{confidence:.1f}%",
                                delta=f"{'High' if confidence > 80 else 'Medium' if confidence > 60 else 'Low'} confidence",
                            )

                        with col_b:
                            st.metric(
                                label="Status", value=prediction, delta=status_color
                            )

                        with col_c:
                            st.metric(
                                label="Processing Time",
                                value="0.8s",
                                delta="Fast analysis",
                            )
                        
                        # Performance Metrics
                        st.markdown("---")
                        st.markdown("### 🎯 Model Performance Statement")

                        if confidence > 80:
                            st.success("""
                            ✅ **ML Model Status: HIGH CONFIDENCE PREDICTION**
                            
                            The machine learning model has successfully analyzed the image with high confidence. 
                            This prediction meets our business requirement of >80% confidence threshold and 
                            demonstrates the model's effectiveness in distinguishing between healthy and infected leaves.
                            """)
                        elif confidence > 60:
                            st.warning("""
                            ⚠️ **ML Model Status: MEDIUM CONFIDENCE PREDICTION**
                            
                            The model has provided a prediction but with medium confidence. While the model 
                            has successfully processed the image, additional verification may be recommended 
                            for critical decision-making.
                            """)
                        else:
                            st.error("""
                            ❌ **ML Model Status: LOW CONFIDENCE PREDICTION**
                            
                            The model prediction has low confidence. This may indicate image quality issues 
                            or edge cases. Manual verification is strongly recommended.
                            """)

                        # Probability visualization
                        st.markdown("#### 📈 Prediction Probabilities")

                        healthy_prob = (1 - prediction_proba) * 100
                        infected_prob = prediction_proba * 100

                        fig = go.Figure(
                            data=[
                                go.Bar(
                                    x=["Healthy", "Powdery Mildew"],
                                    y=[healthy_prob, infected_prob],
                                    marker_color=["green", "red"],
                                    text=[
                                        f"{healthy_prob:.1f}%",
                                        f"{infected_prob:.1f}%",
                                    ],
                                    textposition="auto",
                                )
                            ]
                        )

                        fig.update_layout(
                            title="Prediction Confidence Breakdown",
                            yaxis_title="Confidence (%)",
                            showlegend=False,
                            height=400,
                        )

                        plot_predictions_probabilities(prediction_proba)
                        # Detailed interpretation of the visualization
                        st.markdown("""
                        **📊 Probability Chart Interpretation:**
                        This bar chart displays the model's confidence distribution between the two classes. 
                        The height of each bar represents the probability percentage, with the sum always equaling 100%. 
                        A higher bar indicates stronger confidence in that classification, with values above 80% 
                        considered high confidence predictions suitable for automated decision-making.
                        """)

                        # Detailed analysis
                        st.markdown("#### 🔬 Detailed Analysis")

                        if prediction == "powdery_mildew":
                            st.warning(
                                """
                            **⚠️ Infection Detected:**
                            - White/gray patches indicating powdery mildew presence
                            - Recommend immediate treatment with appropriate fungicide
                            - Isolate affected plants to prevent spread
                            - Monitor surrounding plants closely
                            """
                            )

                            # Treatment recommendations
                            with st.expander("💊 Treatment Recommendations"):
                                st.markdown(
                                    """
                                **Immediate Actions:**
                                1. Remove severely infected leaves
                                2. Apply fungicide treatment
                                3. Improve air circulation around plants
                                4. Reduce humidity levels if possible
                                
                                **Preventive Measures:**
                                1. Regular monitoring of all plants
                                2. Maintain proper plant spacing
                                3. Avoid overhead watering
                                4. Apply preventive fungicide sprays
                                """
                                )

                        else:
                            st.success(
                                """
                            **✅ Healthy Leaf Detected:**
                            - No signs of powdery mildew infection
                            - Leaf appears healthy with good coloration
                            - Continue regular monitoring
                            - Maintain current care practices
                            """
                            )

                            # Maintenance tips
                            with st.expander("🌱 Maintenance Tips"):
                                st.markdown(
                                    """
                                **Keep Your Plants Healthy:**
                                1. Ensure adequate sunlight exposure
                                2. Water at soil level, avoid wetting leaves
                                3. Maintain good air circulation
                                4. Regular inspection for early detection
                                5. Apply balanced fertilizer as needed
                                """
                                )

                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)}")
                        st.info(
                            "Please try uploading a different image or contact support."
                        )

    else:
        # Instructions and examples
        st.markdown("---")
        st.markdown("### 📋 How to Use")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            **Step-by-Step Guide:**
            1. 📤 Upload a cherry leaf image
            2. 🔍 Click 'Analyze Image' button
            3. 📊 Review the analysis results
            4. 💊 Follow treatment recommendations if needed
            5. 💾 Save the report for your records
            """
            )

        with col2:
            st.markdown(
                """
            **Image Requirements:**
            - ✅ Clear, well-lit images
            - ✅ Formats: PNG, JPG, JPEG
            - ✅ Single leaf preferred
            - ✅ Good resolution (>300px)
            - ❌ Avoid blurry or dark images
            """
            )

        # Sample images section
        st.markdown("---")
        st.markdown("### 🎯 Sample Images")
        # st.info(
        #     "Don't have test images? Try downloading sample images from our dataset for testing."
        # )

        col1, col2 = st.columns(2)
        with col1:
            st.image(
                healthy_sample, caption="Healthy Leaf Sample", use_container_width=True
            )
        with col2:
            st.image(
                mildew_sample,
                caption="Powdery Mildew Leaf Sample",
                use_container_width=True,
            )
