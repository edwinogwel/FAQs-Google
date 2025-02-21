### 📢 FAQ Chatbot with Flask & NLP

This is a simple FAQ chatbot built using Flask and Natural Language Processing (NLP). It uses  TF-IDF and cosine similarity to match user queries with predefined FAQs.

### 🚀 Features

✅ Handles common FAQs using machine learning
✅ Built with Flask for easy web integration
✅ Uses TF-IDF for text processing
✅ Simple and lightweight

### 🔧 Setup & Run

1️⃣ Install dependencies:

`pip install flask nltk scikit-learn requests`

2️⃣ Run the Flask app:

`python your_flask_file.py`

3️⃣ Send a request using Python (client.py):

`import requests`

*url = "http://127.0.0.1:5000/chat"
data = {"message": "How do I reset my Google password?"}*

*response = requests.post(url, json=data)
print(response.json())  # {'response': 'You can reset your Google password by going to ...'}*

### 🎯 Future Improvements

  * Add a GUI for better user experience
  * Train with a larger dataset for improved accuracy
  * Implement a chatbot memory for context-aware responses
