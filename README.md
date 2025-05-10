# Chest X-Ray AI Classifier

![Project Banner](https://via.placeholder.com/1200x300.png?text=Chest+X-Ray+AI+Classifier+Banner)  
*Automating Respiratory Disease Detection with Deep Learning*

---

## Overview

The **Chest X-Ray AI Classifier** is a deep learning-based web application designed to assist healthcare professionals in detecting respiratory diseases such as pneumonia and tuberculosis from chest X-ray images. By leveraging artificial intelligence, the system provides automated classifications, user-friendly explanations, and medically informed suggestions, achieving over 85% accuracy on the ChestX-ray14 dataset from Kaggle. The project integrates MobileNetV2 for image classification, HuggingChat for conversational explanations, and a fine-tuned DistilBERT model for generating health suggestions, all deployed through a Flask-based web interface.

---

## Features

- **Image Classification**: Classifies chest X-ray images into three categories: normal, pneumonia, or tuberculosis using MobileNetV2.
- **Conversational Explanations**: Generates user-friendly explanations of predictions using HuggingChat.
- **Medical Suggestions**: Provides actionable health suggestions tailored to the predicted condition using a fine-tuned DistilBERT model.
- **Web Interface**: Deployed as a Flask-based web app, allowing users to upload X-ray images and view results.
- **Ethical Disclaimer**: Includes a disclaimer emphasizing that the system’s outputs are not medical diagnoses and users should consult healthcare professionals.

---

## System Workflow

The application follows a streamlined workflow to process chest X-ray images and deliver comprehensive results:



1. **Upload Image**: Users upload a chest X-ray image via the Flask web app.
2. **Preprocess Image**: The image is resized to 224x224 pixels, converted to grayscale, normalized, and prepared for the MobileNetV2 model.
3. **Classify Image**: MobileNetV2 predicts the class (normal, pneumonia, or tuberculosis) with confidence scores.
4. **Generate Explanation**: HuggingChat provides a conversational explanation of the prediction.
5. **Generate Suggestions**: A fine-tuned DistilBERT model offers medical suggestions based on the predicted condition.
6. **Display Results**: The web app displays the prediction, explanation, and suggestions, along with an ethical disclaimer.

---

## Sample Output

Below is a sample output from the Flask app, showcasing the prediction, explanation, and suggestions for a chest X-ray image:

![Sample Output](output_screenshot.png)

- **Prediction**: Tuberculosis, Confidence: 85.32%
- **Explanation**: "The model identified patterns consistent with tuberculosis, such as increased opacity in the upper lobes. Consult a radiologist for confirmation."
- **Suggestions**: "Maintain a healthy diet rich in vitamins, avoid smoking, ensure proper rest, and monitor for symptoms like persistent cough or fever. Consult a healthcare provider for a comprehensive evaluation."
- **Disclaimer**: "This system is not a substitute for professional medical advice. Please consult a healthcare professional for an official diagnosis."

---

## Dataset

The model was trained and evaluated using the **ChestX-ray14 dataset** from Kaggle, an extension of the ChestX-ray8 dataset. It contains over 112,000 chest X-ray images from more than 30,000 patients, labeled with 14 thoracic diseases. For this project, a subset focusing on three categories—normal, pneumonia, and tuberculosis—was used.

**Reference**:  
Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2097–2106.

The dataset can be accessed on [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) or the [NIH website](https://nihcc.app.box.com/v/ChestXray-NIHCC).

---

## Installation

To set up the Chest X-Ray AI Classifier on your local machine, follow these steps:

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- A GPU (recommended for faster model inference)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/[YourGitHubUsername]/chest-xray-ai-classifier.git
   cd chest-xray-ai-classifier
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` file should include:
   ```
   flask
   tensorflow
   transformers
   opencv-python
   numpy
   requests
   ```

4. **Download Pre-trained Models**:
   - Download the pre-trained MobileNetV2 model and fine-tuned DistilBERT model (not included in the repository due to size constraints).
   - Place them in the `models/` directory:
     - `mobilenetv2_model.h5` (MobileNetV2 model)
     - `distilbert_suggestions/` (fine-tuned DistilBERT model directory)

5. **Set Up HuggingChat API**:
   - Obtain an API key from [Hugging Face](https://huggingface.co/) to use the HuggingChat model.
   - Add the API key to a `.env` file in the project root:
     ```
     HUGGINGFACE_API_KEY=your-api-key-here
     ```
   - Install the `python-dotenv` package to load the API key:
     ```bash
     pip install python-dotenv
     ```

---

## Usage

1. **Run the Flask App**:
   ```bash
   python app.py
   ```
   This will start the web server on `http://localhost:5000`.

2. **Access the Web App**:
   - Open your browser and navigate to `http://localhost:5000`.
   - Upload a chest X-ray image (PNG or JPEG format) using the web interface.

3. **View Results**:
   - The app will display the predicted class, confidence scores, explanation, medical suggestions, and a disclaimer.

**Note**: Ensure the pre-trained models are correctly placed in the `models/` directory, and the HuggingChat API key is configured before running the app.

---

## Project Structure

```
chest-xray-ai-classifier/
├── app.py                   # Main Flask application
├── models/                  # Directory for pre-trained models
│   ├── mobilenetv2_model.h5 # MobileNetV2 model
│   └── distilbert_suggestions/ # Fine-tuned DistilBERT model
├── static/                  # Static files (CSS, JS, images)
│   ├── simple_flowchart.png # Simple workflow chart
│   └── output_screenshot.png # Sample output screenshot
├── templates/               # HTML templates for Flask
│   └── index.html           # Main web page
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (API keys)
└── README.md                # Project documentation
```

---

## Future Improvements

- Retrain the MobileNetV2 model with RGB inputs for improved accuracy.
- Expand the medical suggestions dataset to enhance the DistilBERT model’s outputs.
- Further fine-tune the NLP model for better suggestion quality.
- Deploy the app to a cloud platform (e.g., Heroku, AWS) for broader accessibility.
- Integrate a real-time research API (e.g., Perplexity.ai) for up-to-date medical insights.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit them (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request with a detailed description of your changes.

Please ensure your code follows the project’s coding style and includes appropriate documentation.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, please contact:  
- **Author**: Shubham kumar
- **Email**: shubhamkr2002.in@gmail.com  

---

## Acknowledgments

- The ChestX-ray14 dataset from Kaggle and NIH for providing the data used in this project.
- Hugging Face for providing access to the HuggingChat model and Transformers library.
