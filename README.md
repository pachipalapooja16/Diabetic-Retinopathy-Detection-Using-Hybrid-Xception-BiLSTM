# Advanced Hybrid Deep Learning Approach for Early Detection of Diabetic Retinopathy

## Project Overview

This project presents an end-to-end deep learning approach for the early detection and classification of diabetic retinopathy using retinal fundus images. Diabetic retinopathy is a diabetes-related eye disease that can lead to vision loss if it is not detected and treated at an early stage.

Manual screening of retinal images is time-consuming and depends on the availability of trained ophthalmologists. To support faster and more consistent screening, this project develops a hybrid deep learning model using Xception and BiLSTM.

The proposed model uses Xception for spatial feature extraction and BiLSTM for sequential and contextual feature learning. The final classification layer predicts diabetic retinopathy severity levels from retinal fundus images.

---

## Problem Statement

Existing diabetic retinopathy detection approaches mainly rely on convolutional neural networks for spatial feature extraction. Although CNN-based models can detect important retinal patterns, they may not fully capture contextual relationships between lesion features across severity stages.

This project addresses that limitation by combining:

* Xception for efficient spatial feature extraction
* BiLSTM for contextual and sequential feature learning
* Deep learning-based classification for diabetic retinopathy grading

The main goal is to improve automated diabetic retinopathy detection and support early screening decisions.

---

## Proposed Hybrid Framework

The proposed framework includes dataset preprocessing, Xception-based feature extraction, BiLSTM sequential learning, classification layers, diabetic retinopathy prediction, and performance evaluation.

![Proposed Hybrid Xception BiLSTM Framework](Images/proposed_hybrid_xception_bilstm_framework.png)

---

## Model Architecture

The model combines Xception and BiLSTM in a hybrid architecture. Xception extracts spatial features from retinal fundus images. These extracted features are then passed into a BiLSTM layer to learn contextual relationships. Finally, dense layers and a softmax output layer are used for multi-class diabetic retinopathy classification.

![Xception BiLSTM Architecture](Images/xception_bilstm_architecture.png)

---

## Dataset

The project uses the IDRiD retinal fundus image dataset for diabetic retinopathy classification.

The original dataset contains multiple diabetic retinopathy severity grades. Due to class imbalance, one highly underrepresented class was removed, and the remaining classes were relabelled into four classes for model training.

### Final Class Distribution

| New Class Label | Original Grade | Number of Images |
| --------------- | -------------: | ---------------: |
| 0               |        Grade 0 |              134 |
| 1               |        Grade 2 |              136 |
| 2               |        Grade 3 |               74 |
| 3               |        Grade 4 |               49 |

The raw dataset is not included in this repository due to dataset size and usage restrictions.

---

## Class Distribution

The following chart shows the final diabetic retinopathy class distribution after preprocessing and relabelling.

![Class Distribution](Images/class_distribution.png)

---

## Tools and Technologies Used

* Python
* Google Colab
* TensorFlow
* Keras
* Xception
* BiLSTM
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## Project Workflow

1. Dataset loading and path configuration
2. Data cleaning and label preprocessing
3. Class imbalance handling
4. Image preprocessing and resizing
5. Data augmentation
6. Train-validation split
7. Custom data generator creation
8. Xception-based spatial feature extraction
9. BiLSTM-based sequential feature learning
10. Model training
11. Model evaluation
12. Confusion matrix analysis
13. Sample prediction visualization
14. Result interpretation and conclusion

---

## Model Development

The proposed model uses a hybrid deep learning architecture.

### Xception Feature Extraction

Xception is used as the base model for extracting spatial features from retinal fundus images. It uses depthwise separable convolutions, which help reduce computational complexity while maintaining strong feature extraction capability.

### BiLSTM Sequential Learning

The extracted feature maps from Xception are reshaped into sequential format and passed into a Bidirectional LSTM layer. BiLSTM helps the model learn contextual relationships from both forward and backward directions.

### Classification Layer

The final layers include dense, dropout, and softmax layers. The softmax layer predicts the diabetic retinopathy class.

---

## Training Configuration

| Parameter              |                            Value |
| ---------------------- | -------------------------------: |
| Input Image Size       |                    299 × 299 × 3 |
| Train-Validation Split |                            80:20 |
| Optimizer              |                             Adam |
| Learning Rate          |                           0.0001 |
| Loss Function          | Sparse Categorical Cross-Entropy |
| Epochs                 |                               50 |
| Batch Size             |                               32 |

---

## Model Results

The hybrid Xception–BiLSTM model achieved the following performance:

| Metric              |  Value |
| ------------------- | -----: |
| Training Accuracy   | 96.18% |
| Validation Accuracy | 68.75% |
| AUC Score           | 0.8956 |
| Weighted F1-score   | 0.6692 |

The model achieved moderate validation performance and showed strong learning capability on the training dataset. The difference between training and validation accuracy indicates that further improvements can be made through larger datasets, stronger augmentation, and better imbalance handling.

---

## Classification Report Summary

| Class   | Precision | Recall | F1-score |
| ------- | --------: | -----: | -------: |
| Class 0 |    0.8636 | 0.8636 |   0.8636 |
| Class 1 |    0.6429 | 0.8571 |   0.7347 |
| Class 2 |    0.5000 | 0.3333 |   0.4000 |
| Class 3 |    0.5000 | 0.3333 |   0.4000 |

The model performed better on Class 0 and Class 1. Performance was lower for Class 2 and Class 3 because of class imbalance and visual similarity between advanced diabetic retinopathy severity stages.

---

## Confusion Matrix

The confusion matrix shows how well the model predicted each diabetic retinopathy class compared with the actual class labels.

![Confusion Matrix](Images/confusion_matrix.png)

---

## Sample Predictions

The sample prediction output compares actual diabetic retinopathy labels with predicted labels for selected retinal fundus images.

![Sample Predictions](Images/sample_predictions.png)

---

## Key Findings

* Xception successfully extracted important spatial features from retinal fundus images.
* BiLSTM helped learn contextual relationships from extracted feature sequences.
* Data preprocessing and augmentation improved model training stability.
* The model achieved 68.75% validation accuracy.
* The model performed better on lower diabetic retinopathy classes.
* Higher severity classes showed more misclassification due to limited image samples and class imbalance.
* The hybrid Xception–BiLSTM approach provides a strong foundation for automated diabetic retinopathy screening research.

---

## Limitations

* The dataset size was relatively small.
* The dataset was imbalanced across diabetic retinopathy severity classes.
* External validation on other datasets was not performed.
* The model may not generalize well to all real-world clinical environments without further testing.
* Further improvements are required before clinical deployment.

---

## Future Improvements

* Use a larger and more balanced retinal fundus image dataset.
* Test the model on external datasets such as APTOS, EyePACS, or Messidor.
* Apply advanced data augmentation and oversampling methods.
* Add Grad-CAM explainability to highlight important retinal regions used by the model.
* Compare performance with ResNet50, DenseNet, EfficientNet, and Vision Transformer models.
* Improve deployment readiness by creating a simple web application for demonstration.

---

## Repository Structure

```text
Diabetic-Retinopathy-Detection-Using-Hybrid-Xception-BiLSTM/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── Data/
│   └── README.md
│
├── Images/
│   ├── proposed_hybrid_xception_bilstm_framework.png
│   ├── xception_bilstm_architecture.png
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
│
├── Reports/
│   └── research_report_pooja_pachipala.pdf
│
└── notebooks/
    └── retinopathy_detection_final.ipynb
```

---

## Files Description

| File/Folder        | Description                                         |
| ------------------ | --------------------------------------------------- |
| `README.md`        | Complete project documentation                      |
| `requirements.txt` | Required Python libraries                           |
| `.gitignore`       | Files and folders ignored by Git                    |
| `Data/README.md`   | Dataset information and instructions                |
| `Images/`          | Project diagrams and result visualizations          |
| `Reports/`         | Final research report                               |
| `notebooks/`       | Jupyter notebook containing the full implementation |

---

## How to Run the Project

1. Clone the repository.

```bash
git clone https://github.com/pachipalapooja16/Diabetic-Retinopathy-Detection-Using-Hybrid-Xception-BiLSTM.git
```

2. Install the required libraries.

```bash
pip install -r requirements.txt
```

3. Download the IDRiD dataset from the official source.

4. Update the dataset path in the notebook.

5. Open and run the notebook.

```text
notebooks/retinopathy_detection_final.ipynb
```

---

## Conclusion

This project demonstrates a hybrid Xception–BiLSTM deep learning approach for diabetic retinopathy classification from retinal fundus images. The model combines spatial feature extraction with sequential contextual learning and achieved moderate validation performance.

Although further improvements are required, the project provides a strong foundation for automated diabetic retinopathy detection and medical image classification research.
