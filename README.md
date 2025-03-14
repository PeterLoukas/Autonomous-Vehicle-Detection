# Emergency Vehicle Detection with Deep Learning Techniques for Autonomous Vehicles 🚗🚒🚑

Welcome to the Emergency Vehicle Detection project! This repository outlines a comprehensive approach using Deep Learning to detect emergency vehicles (like police cars, ambulances, fire trucks) for applications in Autonomous Vehicles.

📖 Project Overview
Objective: Enhance autonomous vehicle safety by enabling real-time detection of emergency vehicles in diverse driving conditions.
Problem Addressed: Autonomous vehicles need to recognize and respond to emergency vehicles promptly. Traditional detection systems may struggle with real-world variations like occlusions, lighting, and environmental conditions.
Solution Approach:
Developed a custom dataset with diverse emergency and regular vehicle scenarios.
Leveraged the YOLOv8 (You Only Look Once) deep learning model for efficient and accurate object detection.
Fine-tuned the model using advanced techniques like hyperparameter tuning, transfer learning, and data augmentation to achieve optimal results.
📂 Dataset Creation
1. Sourcing and Frame Extraction
Data Source: Real-world dashcam videos sourced from YouTube.
Extraction Process:
Downloaded videos using yt_dlp.
Extracted every 10th frame to ensure data diversity while reducing redundancy.
Resulted in a collection of 13,953 unique images covering diverse conditions (urban, night, occlusions, blurring).
2. Annotation Process
Automatic Annotation:
Initial annotations were generated using YOLOv8's inference capability.
Manual Refinement:
Used the LabelImg tool for manual review and correction.
Annotation Format:
YOLO Format: cls_id x_center y_center width height (all normalized values).
3. Dataset Composition
Total Images: 13,953
Classes (9 Total):
0: Police Force
1: Emergency Medical Services
2: Car
3: Motorcycle
4: Fire Department
5: Bus
6: Rescue Vehicle
7: Truck
8: Military Vehicle
4. Dataset Splitting
Utilized the autosplit function from Ultralytics:

Training Set: 70% (9,802 images)
Validation Set: 20% (2,823 images)
Test Set: 10% (1,308 images)
Why This Split?
To ensure that the model has sufficient data for training while maintaining reliable sets for validation and unbiased evaluation.

🔍 Exploratory Data Analysis (EDA)
Tools Used:
Ultralytics Explorer API for semantic analysis and SQL-like querying.
LanceDB for similarity search and embedding generation.
Key Insights:
Class Distribution:

Regular vehicles (car, truck, bus) were more frequent compared to emergency vehicles.
Visualized with bar plots and statistical summaries.
Imbalance Handling:

Identified class imbalances using SQL queries.
Removed images with missing or incorrect labels.
Similarity Analysis:

Leveraged LanceDB for embedding-based similarity checks.
Removed near-duplicate images to ensure dataset uniqueness.
🧠 Model Selection and Training
1. Model Comparison (YOLOv8 Variants)
Compared five YOLOv8 variants: Nano (n), Small (s), Medium (m), Large (l), Extra-Large (xl).
Evaluation Metrics: Precision, Recall, mAP (mean Average Precision).
Results were tracked and visualized using Weights & Biases (W&B).
2. Hyperparameter Tuning
Utilized Ray Tune for efficient hyperparameter optimization.
Key Parameters Tuned:
momentum, weight_decay, warmup_epochs, box, cls, dfl
Conducted 4 extensive experiments, each with varied parameter search spaces.
Best hyperparameters were applied in final model training.
3. Fine-Tuning & Transfer Learning
Approach:
Applied transfer learning by freezing the backbone of the YOLO model.
Conducted 10 experiments to find the optimal model setup.
Data Augmentation:
Created a custom augmentation pipeline using Albumentations for:
Noise addition, motion blur, perspective distortion, brightness shifts, and more.
Final Model: The experiment EXP10 demonstrated the highest performance with minimal overfitting.
📊 Results and Evaluation
Evaluation Dataset: Unseen test set (10% split).
Metrics Analyzed:
Precision, Recall, F1-Score, and Loss curves.
Confusion matrices to visualize per-class accuracy.
Key Results:
Consistent and high accuracy in detecting emergency vehicles.
Validated against existing literature benchmarks, demonstrating competitive or superior performance.
🚀 Deployment Approach
Storage: All models, datasets, and results were stored in Google Drive.
Execution: Utilized Google Colab for all code execution.
Inference:
Performed using the final fine-tuned model (best.pt).
Supports inference on both images and videos with configurable thresholds.
⚙️ Usage Instructions
1. Setup
bash
Copy
# Install required dependencies
pip install ultralytics wandb ray[tune] albumentations pytube cv2 supervision
2. Mount Google Drive (for Colab)
python
Copy
from google.colab import drive
drive.mount('/content/drive')
3. Running Inference
python
Copy
from ultralytics import YOLO

# Load the fine-tuned model
model = YOLO('/content/drive/My Drive/Colab Notebooks/Final_Dataset/weights/best.pt')

# Run inference on test images
results = model.predict(
    source='/content/drive/My Drive/Colab Notebooks/Final_Dataset/autosplit_test.txt',
    imgsz=1024,
    conf=0.43,
    device='cpu',
    save=True
)
4. Inference on Videos
python
Copy
# Run inference on a video file
results = model.predict(
    source='/content/drive/My Drive/Colab Notebooks/videos/test_video.mp4',
    save=True,
    save_txt=True,
    conf=0.4
)
🗂️ File Structure
Copy
├── Final_Dataset/
│   ├── images/
│   ├── labels/
│   ├── autosplit_train.txt
│   ├── autosplit_val.txt
│   ├── autosplit_test.txt
│   ├── dataset.yaml
│   └── weights/
│       └── best.pt
├── notebooks/
│   ├── Dataset_Creation.ipynb
│   ├── Dataset_(Pre)processing.ipynb
│   ├── Comparing_YOLOv8_Flavors.ipynb
│   ├── Hyperparameter_Tuning.ipynb
│   ├── Transfer_Learning.ipynb
│   └── Inference_PyTorch.ipynb
✅ Conclusion and Learnings
Key Takeaways:

Successfully developed an advanced YOLOv8-based emergency vehicle detection system.
Efficiently handled dataset imbalances and optimized model training through rigorous experimentation.
Challenges Faced:

Managing class imbalances and ensuring dataset diversity.
Fine-tuning hyperparameters while avoiding overfitting.
Potential Future Improvements:

Enhance real-time inference speed for deployment on edge devices.
Integrate temporal consistency for video-based detection.
Experiment with alternative object detection frameworks.
🤝 Contributions and Support
Feel free to contribute to this project by submitting pull requests or raising issues for improvements. For queries and discussions, please open an issue.

🏁 Acknowledgments
Special thanks to the authors of the referenced academic articles for inspiration.
The Ultralytics team for their powerful YOLOv8 framework.
The developers of Ray Tune, W&B, and LanceDB for their robust tools that supported this research.
