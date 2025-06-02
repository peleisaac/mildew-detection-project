import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import os
from PIL import Image
from src.data_management import load_pkl_file


def page_ml_performance_body():
    """
    Display ML model performance metrics and evaluation results
    Business Requirement 2: Model performance evaluation for cherry leaf classification
    """

    st.write("### 🤖 Machine Learning Model Performance")

    st.info(
        "**Business Requirement 2**: The client is interested in predicting if a cherry tree is healthy "
        "or contains powdery mildew. This page demonstrates the model's capability to meet this requirement."
    )

    # Model Overview Section
    st.write("---")
    st.write("## 📋 Model Overview")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
        **Model Architecture**: Convolutional Neural Network (CNN)
        
        **Key Specifications**:
        - **Task**: Binary Image Classification
        - **Input**: RGB Images (256x256 pixels)
        - **Output**: Healthy vs Powdery Mildew prediction
        - **Architecture**: Custom CNN with 3 Conv layers + Dense layers
        - **Optimizer**: Adam
        - **Loss Function**: Binary Crossentropy
        - **Activation**: ReLU (hidden), Sigmoid (output)
        """
        )

    with col2:
        # Display key performance metrics at a glance
        try:
            performance_data = load_evaluation_data()
            if performance_data and "performance_summary" in performance_data:
                perf = performance_data["performance_summary"]
                st.metric("Test Accuracy", f"{perf.get('accuracy', 0):.2%}")
                st.metric("Precision", f"{perf.get('precision', 0):.2%}")
                st.metric("Recall", f"{perf.get('recall', 0):.2%}")
                st.metric("F1-Score", f"{perf.get('f1_score', 0):.2%}")
        except Exception as e:
            st.warning(
                "Performance metrics not available. Please run model evaluation notebook."
            )

    # Training History Section
    st.write("---")
    st.write("## 📈 Training Performance")

    display_training_curves()

    # Model Evaluation Metrics
    st.write("---")
    st.write("## 📊 Model Evaluation Metrics")

    # Performance Summary
    display_performance_summary()

    # Confusion Matrix
    st.write("### 🎯 Confusion Matrix")
    display_confusion_matrix()

    # Classification Report
    st.write("### 📋 Classification Report")
    display_classification_report()

    # ROC Curve
    st.write("### 📈 ROC Curve Analysis")
    display_roc_curve()

    # Model Predictions on Test Set
    st.write("---")
    st.write("## 🔍 Sample Predictions")
    display_sample_predictions()

    # Model Architecture Details
    st.write("---")
    st.write("## 🏗️ Model Architecture")
    display_model_architecture()

    # Business Impact Assessment
    st.write("---")
    st.write("## 💼 Business Impact Assessment")
    display_business_impact()


@st.cache_data
def load_evaluation_data():
    """Load evaluation data from pickle file"""
    try:
        return load_pkl_file("outputs/evaluation.pkl")
    except:
        # Fallback to individual JSON files if pickle not available
        try:
            evaluation_data = {}

            # Load performance summary
            if os.path.exists("outputs/performance_summary.json"):
                with open("outputs/performance_summary.json", "r") as f:
                    evaluation_data["performance_summary"] = json.load(f)

            # Load classification report
            if os.path.exists("outputs/classification_report.json"):
                with open("outputs/classification_report.json", "r") as f:
                    evaluation_data["classification_report"] = json.load(f)

            # Load confusion matrix
            if os.path.exists("outputs/confusion_matrix.json"):
                with open("outputs/confusion_matrix.json", "r") as f:
                    evaluation_data["confusion_matrix"] = json.load(f)

            # Load ROC data
            if os.path.exists("outputs/roc_data.json"):
                with open("outputs/roc_data.json", "r") as f:
                    evaluation_data["roc_data"] = json.load(f)

            # Load training history
            if os.path.exists("outputs/history.json"):
                with open("outputs/history.json", "r") as f:
                    evaluation_data["training_history"] = json.load(f)

            return evaluation_data
        except Exception as e:
            st.error(f"Could not load evaluation data: {str(e)}")
            return None


def display_training_curves():
    """Display training and validation accuracy/loss curves"""
    try:
        # Try to load training history plots
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Training & Validation Accuracy")
            accuracy_plot_path = "outputs/plots/model_training_accuracy.png"
            if os.path.exists(accuracy_plot_path):
                accuracy_img = Image.open(accuracy_plot_path)
                st.image(accuracy_img, use_container_width=True)
                st.success(
                    """
                **Accuracy Analysis:**
                - Model shows strong learning progression
                - Training and validation curves converge well
                - No significant overfitting observed
                - Stable performance achieved
                """
                )
            else:
                st.warning("Training accuracy plot not found.")

        with col2:
            st.markdown("#### Training & Validation Loss")
            loss_plot_path = "outputs/plots/model_training_losses.png"
            if os.path.exists(loss_plot_path):
                loss_img = Image.open(loss_plot_path)
                st.image(loss_img, use_container_width=True)
                st.info(
                    """
                **Loss Analysis:**
                - Consistent loss reduction during training
                - Validation loss follows training trend
                - Model convergence achieved
                - Good generalization indicated
                """
                )
            else:
                st.warning("Training loss plot not found.")

    except Exception as e:
        st.error(f"Error displaying training curves: {str(e)}")


def display_performance_summary():
    """Display model performance summary metrics"""
    try:
        evaluation_data = load_evaluation_data()
        if not evaluation_data or "performance_summary" not in evaluation_data:
            st.warning("Performance summary not available.")
            return

        perf = evaluation_data["performance_summary"]

        st.markdown("#### 🎯 Key Performance Indicators")

        # Create metrics in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            accuracy = perf.get("accuracy", 0)
            st.metric(
                label="Accuracy",
                value=f"{accuracy:.1%}",
                help="Overall correctness of predictions",
            )

        with col2:
            precision = perf.get("precision", 0)
            st.metric(
                label="Precision",
                value=f"{precision:.1%}",
                help="Proportion of positive predictions that were correct",
            )

        with col3:
            recall = perf.get("recall", 0)
            st.metric(
                label="Recall",
                value=f"{recall:.1%}",
                help="Proportion of actual positives correctly identified",
            )

        with col4:
            f1_score = perf.get("f1_score", 0)
            st.metric(
                label="F1-Score",
                value=f"{f1_score:.1%}",
                help="Harmonic mean of precision and recall",
            )

        # Performance interpretation
        if accuracy >= 0.97:
            st.success(
                "🌟 **Excellent Performance**: Model exceeds business requirements with outstanding accuracy."
            )
        elif accuracy >= 0.90:
            st.success(
                "✅ **Strong Performance**: Model meets business requirements with high accuracy."
            )
        elif accuracy >= 0.70:
            st.info(
                "⚠️ **Acceptable Performance**: Model meets minimum business requirements."
            )
        else:
            st.error(
                "❌ **Below Requirements**: Model does not meet minimum accuracy threshold of 70%."
            )

    except Exception as e:
        st.error(f"Error displaying performance summary: {str(e)}")


def display_confusion_matrix():
    """Display confusion matrix with interpretation"""
    try:
        evaluation_data = load_evaluation_data()
        if not evaluation_data:
            st.warning("Confusion matrix data not available.")
            return

        # Try to load confusion matrix plot first
        cm_plot_path = "outputs/plots/confusion_matrix.png"
        if os.path.exists(cm_plot_path):
            cm_img = Image.open(cm_plot_path)
            st.image(cm_img, use_container_width=True)
        else:
            # Generate confusion matrix if data is available
            cm_data = evaluation_data.get("confusion_matrix")
            if cm_data:
                if isinstance(cm_data, list):
                    cm = np.array(cm_data)
                else:
                    cm = cm_data

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    ax=ax,
                    xticklabels=["Healthy", "Powdery Mildew"],
                    yticklabels=["Healthy", "Powdery Mildew"],
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)
                plt.close()

        # Interpretation
        if "confusion_matrix" in evaluation_data:
            cm = evaluation_data["confusion_matrix"]
            if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2:
                tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"""
                    **Confusion Matrix Values:**
                    - True Negatives (Healthy correctly identified): {tn}
                    - False Positives (Healthy misclassified as Mildew): {fp}
                    - False Negatives (Mildew misclassified as Healthy): {fn}
                    - True Positives (Mildew correctly identified): {tp}
                    """
                    )

                with col2:
                    if fn > fp:
                        st.warning(
                            "⚠️ **Model tends to miss some mildew cases** - Consider adjusting threshold for agricultural safety."
                        )
                    elif fp > fn:
                        st.info(
                            "ℹ️ **Model is conservative** - May flag some healthy leaves as infected."
                        )
                    else:
                        st.success("✅ **Balanced performance** across both classes.")

    except Exception as e:
        st.error(f"Error displaying confusion matrix: {str(e)}")


def display_classification_report():
    """Display detailed classification report"""
    try:
        evaluation_data = load_evaluation_data()
        if not evaluation_data or "classification_report" not in evaluation_data:
            st.warning("Classification report not available.")
            return

        report_data = evaluation_data["classification_report"]

        # Convert to DataFrame for better display
        if isinstance(report_data, dict):
            # Remove 'accuracy' key if present for cleaner display
            display_data = {
                k: v
                for k, v in report_data.items()
                if k != "accuracy" and isinstance(v, dict)
            }

            df = pd.DataFrame(display_data).T
            df = df.round(3)

            # Style the dataframe
            styled_df = df.style.format(
                {
                    "precision": "{:.3f}",
                    "recall": "{:.3f}",
                    "f1-score": "{:.3f}",
                    "support": "{:.0f}",
                }
            ).background_gradient(
                subset=["precision", "recall", "f1-score"],
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
            )

            st.dataframe(styled_df, use_container_width=True)

            # Insights
            if "healthy" in display_data and "powdery_mildew" in display_data:
                healthy_f1 = display_data["healthy"].get("f1-score", 0)
                mildew_f1 = display_data["powdery_mildew"].get("f1-score", 0)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Healthy F1-Score", f"{healthy_f1:.3f}")
                with col2:
                    st.metric("Mildew F1-Score", f"{mildew_f1:.3f}")

                if abs(healthy_f1 - mildew_f1) < 0.05:
                    st.success(
                        "✅ **Balanced Performance**: Model performs equally well on both classes."
                    )
                else:
                    better_class = "Healthy" if healthy_f1 > mildew_f1 else "Mildew"
                    st.info(
                        f"ℹ️ **Slight Class Preference**: Model performs slightly better on {better_class} detection."
                    )
        else:
            st.json(report_data, expanded=False)

    except Exception as e:
        st.error(f"Error displaying classification report: {str(e)}")


def display_roc_curve():
    """Display ROC curve and AUC analysis"""
    try:
        # Try to load ROC curve plot
        roc_plot_path = "outputs/plots/roc_curve.png"
        if os.path.exists(roc_plot_path):
            roc_img = Image.open(roc_plot_path)
            st.image(roc_img, use_container_width=True)
        else:
            # Generate ROC curve if data is available
            evaluation_data = load_evaluation_data()
            if evaluation_data and "roc_data" in evaluation_data:
                roc_data = evaluation_data["roc_data"]
                fpr = roc_data.get("fpr", [])
                tpr = roc_data.get("tpr", [])
                auc_score = roc_data.get("auc", 0)

                if fpr and tpr:
                    fig = go.Figure()

                    # ROC Curve
                    fig.add_trace(
                        go.Scatter(
                            x=fpr,
                            y=tpr,
                            mode="lines",
                            name=f"ROC Curve (AUC = {auc_score:.3f})",
                            line=dict(color="darkorange", width=2),
                        )
                    )

                    # Diagonal line
                    fig.add_trace(
                        go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            mode="lines",
                            name="Random Classifier",
                            line=dict(color="navy", width=2, dash="dash"),
                        )
                    )

                    fig.update_layout(
                        title="Receiver Operating Characteristic (ROC) Curve",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        width=600,
                        height=500,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # AUC interpretation
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("AUC Score", f"{auc_score:.3f}")

                    with col2:
                        if auc_score >= 0.95:
                            st.success("🌟 Excellent discriminative ability")
                        elif auc_score >= 0.90:
                            st.success("✅ Very good discriminative ability")
                        elif auc_score >= 0.80:
                            st.info("👍 Good discriminative ability")
                        else:
                            st.warning("⚠️ Fair discriminative ability")

        # ROC interpretation
        st.markdown(
            """
        **ROC Curve Interpretation:**
        - **AUC = 1.0**: Perfect classifier
        - **AUC > 0.9**: Excellent performance
        - **AUC > 0.8**: Good performance  
        - **AUC = 0.5**: Random classifier (diagonal line)
        - **Closer to top-left corner**: Better performance
        """
        )

    except Exception as e:
        st.error(f"Error displaying ROC curve: {str(e)}")


def display_sample_predictions():
    """Display sample predictions from test set"""
    try:
        predictions_plot_path = "outputs/plots/sample_predictions.png"
        if os.path.exists(predictions_plot_path):
            st.markdown("#### 🎯 Model Predictions on Test Samples")
            predictions_img = Image.open(predictions_plot_path)
            st.image(
                predictions_img,
                use_container_width=True,
                caption="Sample predictions showing model confidence and accuracy on test data",
            )
            st.info(
                """
            **Sample Predictions Analysis:**
            - Green borders indicate correct predictions
            - Red borders indicate incorrect predictions  
            - Confidence scores show model certainty
            - Diverse samples demonstrate robustness
            """
            )
        else:
            st.info(
                "Sample predictions visualization not available. Generate using model evaluation notebook."
            )

    except Exception as e:
        st.error(f"Error displaying sample predictions: {str(e)}")


def display_model_architecture():
    """Display model architecture and complexity"""
    try:
        # Try to load model summary
        model_summary_path = "outputs/model_summary.json"
        if os.path.exists(model_summary_path):
            with open(model_summary_path, "r") as f:
                model_summary = json.load(f)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🏗️ Architecture Summary")
                st.json(model_summary, expanded=True)

            with col2:
                st.markdown("#### 📊 Model Complexity")
                if "total_params" in model_summary:
                    st.metric("Total Parameters", f"{model_summary['total_params']:,}")
                if "trainable_params" in model_summary:
                    st.metric(
                        "Trainable Parameters", f"{model_summary['trainable_params']:,}"
                    )
                if "model_size_mb" in model_summary:
                    st.metric("Model Size", f"{model_summary['model_size_mb']:.1f} MB")
        else:
            st.info("Model architecture summary not available.")

    except Exception as e:
        st.error(f"Error displaying model architecture: {str(e)}")


def display_business_impact():
    """Display business impact assessment and requirements validation"""
    st.markdown(
        """
    ### 🎯 Business Requirements Validation
    
    **Requirement**: Achieve minimum 70% accuracy for reliable mildew detection
    """
    )

    try:
        evaluation_data = load_evaluation_data()
        if evaluation_data and "performance_summary" in evaluation_data:
            accuracy = evaluation_data["performance_summary"].get("accuracy", 0)

            col1, col2 = st.columns([1, 2])

            with col1:
                if accuracy >= 0.70:
                    st.success(f"✅ **REQUIREMENT MET**\nAccuracy: {accuracy:.1%}")
                else:
                    st.error(f"❌ **REQUIREMENT NOT MET**\nAccuracy: {accuracy:.1%}")

            with col2:
                # Calculate business impact metrics
                if accuracy >= 0.95:
                    time_savings = "90-95%"
                    reliability = "Excellent"
                elif accuracy >= 0.90:
                    time_savings = "80-90%"
                    reliability = "Very Good"
                elif accuracy >= 0.70:
                    time_savings = "60-80%"
                    reliability = "Good"
                else:
                    time_savings = "<60%"
                    reliability = "Poor"

                st.markdown(
                    f"""
                **Projected Business Impact:**
                - **Time Savings**: {time_savings} reduction in manual inspection
                - **Reliability**: {reliability} automated detection
                - **Scalability**: Thousands of trees can be processed daily
                - **Cost Efficiency**: Significant reduction in labor costs
                """
                )
    except:
        st.info("Business impact assessment requires performance data.")

    # Implementation recommendations
    st.markdown(
        """
    ### 💡 Implementation Recommendations
    
    **Deployment Strategy:**
    - Deploy model in production agricultural environment
    - Implement mobile app for field use
    - Set up batch processing for large-scale monitoring
    - Establish feedback loop for continuous improvement
    
    **Risk Mitigation:**
    - Regular model retraining with new data
    - Human expert validation for critical decisions
    - Confidence threshold adjustment based on use case
    - Backup manual inspection protocols
    """
    )
