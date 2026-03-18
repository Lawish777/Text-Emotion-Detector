# Text Emotion Detector 🧠

A sleek, interactive web app built with **Streamlit** that detects emotions in text using a pre-trained machine learning model.

## Features

- Real-time emotion prediction from any input text
- Displays the top predicted emotion with matching emoji
- Shows confidence score and top-3 probability breakdown
- Clean, modern UI with horizontal bar chart visualization (Altair)
- Input validation, character counter & loading spinner
- Sidebar with model info and usage tips
- Mobile-friendly wide layout

## Supported Emotions

- 😠 Anger  
- 🤮 Disgust  
- 😨 Fear  
- 😊 Happy  
- 😂 Joy  
- 😐 Neutral  
- 😔 Sad / Sadness  
- 😳 Shame  
- 😮 Surprise  


## Tech Stack

| Component          | Technology used                  |
|--------------------|----------------------------------|
| Web Framework      | Streamlit                        |
| Machine Learning   | scikit-learn (Logistic Regression) |
| Vectorization      | TF-IDF                           |
| Model Persistence  | joblib                           |
| Data Handling      | pandas, numpy                    |
| Visualization      | Altair                           |
| UI/UX Enhancements | Custom layout, spinner, sidebar  |
