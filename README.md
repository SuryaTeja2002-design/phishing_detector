# PhishGuard AI – Machine Learning Phishing URL Detector

PhishGuard AI is a machine learning–based phishing detection system that analyzes URLs and predicts whether they are malicious or legitimate. The system extracts lexical, domain, and behavioral features from URLs and evaluates them using multiple machine learning models to estimate phishing probability.

The project also includes an interactive web interface that visualizes model predictions, risk scores, and detection metrics to help users understand the analysis results.

---

## Features

• Detects phishing URLs using machine learning
• Extracts over 30 lexical and structural features from URLs
• Supports multiple ML models for prediction
• Displays phishing probability and cyber risk score
• Multi-model comparison visualization
• Confidence and uncertainty analysis
• Interactive web interface for real-time URL scanning

---

## Machine Learning Models Used

The system evaluates URLs using multiple classifiers:

• Gradient Boosting
• Random Forest
• Support Vector Machine (RBF Kernel)
• Multi-Layer Perceptron Neural Network

Predictions from these models are compared to generate a consensus prediction and confidence score.

---

## Feature Engineering

The system extracts several types of features from URLs, including:

• URL length and structure
• Presence of suspicious keywords (login, verify, secure, etc.)
• Subdomain analysis
• Domain entropy
• IP-based URLs
• Use of URL shortening services
• Suspicious path patterns

These features help identify common phishing patterns used in malicious links.

---

## Project Structure

```
phishing_detector/
│
├── app.py                # Flask web application
├── feature_extractor.py  # URL feature extraction
├── train_model.py        # ML model training
├── predict_cli.py        # Command line prediction tool
├── cyber_analysis.py     # Risk scoring and analysis
├── online_learner.py     # Model improvement logic
├── model.pkl             # Trained ML model
├── templates/            # HTML interface
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/phishing_detector.git
cd phishing_detector
```

Install dependencies:

```
pip install -r requirements.txt
```

---
## Running the model training

Train the model:

```
python train_model.py
```

## Running the Web Application

Start the Flask server:

```
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

Enter a URL and click **Scan URL** to analyze it.

---

## Example Detection

Input URL:

```
http://microsoft-security-update.com/credential-reset
```

Output:

```
Phishing Detected
Probability: 88%
Risk Level: High
```

The system also displays model comparison results and confidence metrics.

---

## Dataset Sources

Phishing datasets can be obtained from public threat intelligence feeds such as:

• PhishTank
• OpenPhish

These datasets are commonly used for phishing detection research.

---

## Technologies Used

• Python
• Scikit-Learn
• Flask
• HTML / CSS / JavaScript
• Machine Learning Classification Models

---

## Future Improvements

• Real-time phishing feed integration
• Browser extension for automatic URL scanning
• Email phishing detection using NLP
• Continuous online learning for model updates

---

## Author

Surya Teja Gorthi

---

## License

This project is intended for educational and research purposes.

Images with the tested results

<img width="1244" height="683" alt="image" src="https://github.com/user-attachments/assets/78611d85-d983-48f3-816e-faa0394c9cb7" />

<img width="1073" height="443" alt="image" src="https://github.com/user-attachments/assets/b3e62734-c47f-498a-afaa-b20aae731528" />




