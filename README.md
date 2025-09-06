🦴 Joint Disease Detection and Severity Prediction

AI-driven system for detecting multiple joint disorders from X-rays and predicting severity levels for Knee Osteoarthritis (KOA).
Built with EfficientNet, Grad-CAM explainability, and a Flask web interface for interactive testing.

✨ Features

Detects multiple joint disorders, including:

Knee Osteoarthritis (KOA)
Rheumatoid Arthritis
Fractures

Severity grading for KOA:

KL Grade 0 → No OA
KL Grade 1 → Doubtful OA
KL Grade 2 → Mild OA
KL Grade 3 → Moderate OA
KL Grade 4 → Severe OA

Grad-CAM heatmaps for visual explainability of predictions

Flask web app for uploading X-rays and viewing results

Clean, modular, and scalable training and inference pipelines

🧰 Tech Stack

Python
TensorFlow / Keras
OpenCV / NumPy / Pandas
Flask
Matplotlib / Seaborn

🚀 QuickStart

# Clone the repository
git clone https://github.com/divyaboddepalli/Joint-disease-detection-and-severity-prediction.git
cd joint-disease-detection-and-severity-prediction

# Install dependencies
pip install -r requirements.txt

# Run the web app
python app.py

📁 Dataset & Models

Datasets: Sample X-ray datasets included. Full datasets stored externally due to large size.

Pre-trained Models: EfficientNet-based models for KOA severity prediction and multi-disease detection included.

Large Files: Files exceeding GitHub limits (50–100 MB) are stored using Git LFS or external storage.

🖼️ Demo

Upload an X-ray or joint scan via the web interface.

View predicted joint disorder(s).

Get KOA severity grading (if applicable).

Examine Grad-CAM heatmaps highlighting affected regions.

📝 Why This Project Stands Out

This project highlights expertise in end-to-end AI for medical imaging, from data preprocessing and deep learning model training to interactive web deployment:

Medical Image Analysis – Strong experience with X-ray-based multi-disease detection and KOA severity prediction.

Explainable AI – Grad-CAM visualizations make predictions interpretable and clinically relevant.

Web Integration – Flask app demonstrates deployment skills and real-time testing capabilities.

Code Quality & Modularity – Structured, reusable code suitable for research or production extension.

Healthcare Problem-Solving – Addresses real-world medical imaging challenges, translating AI insights into actionable results.

This project is ideal for AI, deep learning, medical imaging, and healthcare tech roles.
