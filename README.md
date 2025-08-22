# Credit Card Fraud Detection

A comprehensive machine learning project that implements multiple algorithms to detect fraudulent credit card transactions. This project includes both a Jupyter notebook for model development and a Streamlit web application for real-time predictions.

## 🚀 Features

- **Multiple ML Algorithms**: Implements Logistic Regression, Random Forest, and XGBoost classifiers
- **Interactive Web Interface**: Streamlit-based web application for easy model selection and predictions
- **Data Upload & Processing**: Support for CSV file uploads with automatic preprocessing
- **Model Performance Comparison**: Comprehensive evaluation metrics including accuracy, precision, recall, and F1-score
- **Export Functionality**: Download predictions as CSV files
- **Pre-trained Models**: Includes saved models for immediate use

## 📊 Model Performance

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| **Logistic Regression** | 95.0% | 96.0% | 95.0% | 95.0% |
| **Random Forest** | 95.4% | 96.0% | 95.0% | 95.0% |
| **XGBoost** | 95.9% | 96.0% | 96.0% | 96.0% |

## 🛠️ Technologies Used

- **Python 3.x**
- **Machine Learning**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy
- **Web Framework**: Streamlit
- **Data Visualization**: seaborn, matplotlib

## 📁 Project Structure

```
Credit-card-fraud-detection/
├── app.py                          # Streamlit web application
├── Credit card Fraud Detection.ipynb  # Jupyter notebook with model development
├── creditcard.csv                  # Training dataset
├── testing dataset.csv             # Test dataset
├── logistic_model.pkl             # Saved Logistic Regression model
├── randomforest.pkl               # Saved Random Forest model
├── xgboost_model.pkl              # Saved XGBoost model
├── requirements.txt                # Python dependencies
└── README.md                      # This file
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Credit-card-fraud-detection.git
   cd Credit-card-fraud-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web application**
   ```bash
   streamlit run app.py
   ```

## 📖 Usage

### Web Application

1. **Launch the app**: Run `streamlit run app.py`
2. **Select Model**: Choose from Logistic Regression, Random Forest, or XGBoost
3. **Upload Data**: Upload a CSV file containing transaction data
4. **View Results**: See predictions and download results as CSV

### Jupyter Notebook

1. Open `Credit card Fraud Detection.ipynb` in Jupyter
2. Run cells sequentially to understand the data preprocessing and model training
3. Experiment with different algorithms and parameters

## 🔍 Data Format

The application expects CSV files with the following structure:
- **Features**: V1, V2, ..., V28 (anonymized transaction features)
- **Target**: Class (0 for legitimate, 1 for fraudulent transactions)

**Note**: The V1-V28 features are anonymized for privacy and security reasons, as credit card data contains sensitive information.

## 🧠 Model Details

### Data Preprocessing
- **Handling Imbalanced Data**: Uses undersampling technique to balance legitimate vs. fraudulent transactions
- **Feature Engineering**: Works with 28 anonymized features (V1-V28)
- **Data Validation**: Automatic handling of missing values and data type conversion

### Algorithms Implemented

1. **Logistic Regression**
   - Baseline model for binary classification
   - Good interpretability and fast training

2. **Random Forest**
   - Ensemble method with 200 estimators
   - Balanced class weights for handling imbalanced data

3. **XGBoost**
   - Gradient boosting algorithm
   - Optimized hyperparameters for fraud detection

## 📈 Performance Metrics

The models are evaluated using:
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

## 🔧 Customization

### Adding New Models
1. Train your model in the Jupyter notebook
2. Save it using pickle: `pickle.dump(model, open("your_model.pkl", "wb"))`
3. Add it to the `models` dictionary in `app.py`

### Modifying Features
- Update the feature preprocessing in the notebook
- Ensure the web app handles the new feature structure
- Retrain and save updated models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Dataset: Credit card fraud detection dataset
- Libraries: scikit-learn, XGBoost, Streamlit, pandas, numpy
- Community: Open source machine learning community

## 📞 Support

If you have any questions or need help with the project:
- Open an issue on GitHub
- Check the Jupyter notebook for detailed implementation
- Review the Streamlit app code for usage examples

---

**Note**: This project is for educational and research purposes. Always ensure compliance with data privacy regulations when working with financial data.
