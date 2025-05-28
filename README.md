# 🌿 SmartLeaf: Mildew Detection in Cherry Leaves

## 📌 Project Summary

SmartLeaf is a machine learning-powered solution designed to automatically classify cherry leaves as **healthy** or affected by **powdery mildew**. This tool provides real-time predictions and visual analytics via an interactive dashboard to support plant pathologists, farmers, and agricultural scientists in rapid diagnosis and intervention.

---

## 🎯 Business Case & Objective

We are building an ML solution to address the following key business objective:

> **"Predict whether a given cherry leaf image is healthy or affected by powdery mildew."**

### Client Needs

* A **dashboard** that:

  * Accepts user-uploaded cherry leaf images
  * Provides real-time predictions
  * Visualizes key image analysis insights
  * Tracks analysis history and health distribution statistics

---

## 💡 Pre-Project Exploration & Understanding

Before development began, the following questions were asked to understand the project's full context:

### Q\&A Framework

1. **What is the business objective requiring an ML solution?**

   * To predict whether a given cherry leaf is healthy or has powdery mildew.

2. **Is data available for model training?**

   * Yes. A labeled image dataset with healthy and powdery mildew-affected leaves.

3. **What type of solution is needed?**

   * A **Streamlit dashboard** (not an API) to make predictions and show visual analytics.

4. **What does success look like?**

   * A user-friendly dashboard that meets the following:

     * Accepts user-uploaded images
     * Displays visual differentiations between classes
     * Returns confident predictions with metadata
     * Maintains a visual analysis history

5. **What are the Epics and User Stories?**

   * **Epic 1: Data Collection & Understanding**

     * Story: Gather cherry leaf images
     * Story: Validate data integrity
   * **Epic 2: Data Preprocessing & Visualization**

     * Story: Clean corrupted images
     * Story: Visualize class distribution and image features
   * **Epic 3: Model Development & Evaluation**

     * Story: Train CNN for binary classification
     * Story: Plot learning curves and test accuracy
   * **Epic 4: Dashboard Design & Deployment**

     * Story: Develop user interface with Streamlit
     * Story: Deploy with persistent image history

6. **Are there ethical or privacy concerns?**

   * Minimal. As long as uploaded leaf images do not include identifying data.

7. **What level of model performance is needed?**

   * At least **70% accuracy** to be considered useful. Final model exceeds 99%.

8. **What are the inputs and outputs?**

   * **Input**: Cherry leaf image (unlabeled)
   * **Output**: Health status (healthy / mildew) and confidence score

9. **Does the data suggest a specific model?**

   * Yes. Image data + binary classification = Convolutional Neural Network (CNN)

10. **How does this benefit the client?**

    * Faster, more scalable mildew detection
    * Reduction in manual diagnosis
    * Supports proactive plant care

---

## 📂 Project Structure

The project is divided into three main notebooks:

### 1. 📚 Data Collection

* Downloaded the dataset from Kaggle using the Kaggle API and JSON credentials
* Extracted the zip file using Python's `zipfile` module

```python
with zipfile.ZipFile("inputs/cherry_leaves/cherry-leaves.zip", 'r') as zip_ref:
    zip_ref.extractall("inputs/cherry_leaves/")
```

* Verified image validity using PIL to detect corrupt or unreadable files
* Split the dataset into `train`, `validation`, and `test` subsets using a 70/20/10 ratio
* Used `organize_dataset_into_subsets()` and saved results into `inputs/split-leaves` to preserve the original dataset

### 2. 🌐 Data Visualization

* Defined base dataset path and structure using `Path`
* Counted images per class and subset
* Displayed sample image grids for both `healthy` and `powdery_mildew` classes
* Analyzed image dimensions (found consistent size: **256x256**)
* Plotted image shape scatterplot and histograms
* Computed and displayed average image per class
* Visualized the **absolute pixel difference** between healthy and mildew images

### 3. 🧑‍💻 Data Evaluation

* Built a CNN model with three convolutional layers, followed by a dense layer
* Used `ImageDataGenerator` for image augmentation (rotation, flipping, zooming)
* Trained model with early stopping and model checkpointing
* Achieved over **99% accuracy** on training and validation sets
* Plotted model learning curves (accuracy & loss) and saved them to the `outputs/` folder
* Evaluated the model using a confusion matrix and classification report
* Visualized predictions for random test images
* Predicted on 3 random test images with actual-vs-predicted label and confidence scores displayed, simulating dashboard behavior. This helped verify model reliability and interpretability.

![Screenshot of Prediction](https://github.com/peleisaac/mildew-detection-project/blob/main/images/predict_on_random_test_data.png)

---

## 📸 Is a Montage Needed?

While a traditional "montage" isn't necessary, we used **image grids and comparative visualizations** to:

* Display representative images from each class
* Visually explain what the model is learning
* Compare average image patterns

These are more practical and tailored to machine learning evaluation than a static montage.

---

## 🌎 Dashboard Design Plan

To fulfill business and user experience goals, the dashboard is structured with tabbed navigation using Streamlit. Here's the planned layout:

### Tabs:

#### 🔍 Predict

* Upload a cherry leaf image
* Display uploaded image preview
* Predict class (healthy / powdery\_mildew)
* Show confidence score visually

#### 📊 Visual Difference

* Display **average image for healthy leaves**
* Display **average image for mildew leaves**
* Display **absolute pixel difference image**
* Optional: include sample comparisons and short textual cues on what the user is seeing

#### ℹ️ About

* App purpose and use case
* Model accuracy and summary
* Project contributors and credits

This design aims to combine prediction capability with educational insights, providing transparency and enhancing trust in the model.

---


