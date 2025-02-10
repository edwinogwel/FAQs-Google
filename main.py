import string
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify
nltk.download("punkt")
nltk.download("punkt_tab")

app = Flask(__name__)

# FAQ dataset
faq_pairs = {
    "What is your name?": "I am a chatbot designed to answer FAQs.",
    "How can I contact support?": "You can contact support via email at support@example.com.",
    "What are your working hours?": "We are available 24/7 to assist you.",
    "What is AI?": "AI stands for Artificial Intelligence, which enables machines to mimic human intelligence.",
    "How do I reset my Google password?": "You can reset your Google password by going to https://accounts.google.com/signin/recovery.",
    "How do I recover my Gmail account?": "Visit Google's account recovery page and follow the instructions to recover your Gmail account.",
    "How do I enable two-factor authentication on Google?": "Go to your Google Account security settings and enable two-factor authentication for added security.",
    "How do I free up space in Google Drive?": "You can delete large files, empty the trash, or upgrade your storage plan in Google Drive settings.",
    "How do I share a file on Google Drive?": "Open the file in Google Drive, click on 'Share,' and enter the email addresses of the people you want to share it with.",
    "How do I use Google Maps offline?": "Open Google Maps, search for a location, and select 'Download offline map' to access it without the internet.",
    "How do I report a problem on Google Maps?": "Click on 'Send Feedback' in Google Maps and describe the issue for Google to review.",
    "How do I change my Google Search settings?": "Go to Google Search settings and modify preferences like SafeSearch, region settings, and search history.",
    "How do I delete my Google search history?": "Visit My Activity on Google and delete your search history from your account settings.",
    "How do I clear cache and cookies in Chrome?": "Go to Chrome settings, select 'Privacy and security,' then 'Clear browsing data' to remove cache and cookies.",
    "How do I enable dark mode in Google Chrome?": "Go to Chrome settings, select 'Appearance,' and choose 'Dark theme' to enable dark mode.",
    "How do I create a new Google account?": "Visit https://accounts.google.com/signup and follow the steps to create a new Google account.",
    "How do I delete my Google account?": "Go to Google Account settings, select 'Data & personalization,' and choose 'Delete your Google Account.'"
}

corpus = list(faq_pairs.keys())        # Questions
responses = list(faq_pairs.values())


def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, token_pattern=None)
X = vectorizer.fit_transform(corpus)


def get_response(user_input):
    user_input = preprocess(user_input)
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, X)
    best_match_idx = np.argmax(similarities)
    if similarities[0, best_match_idx] < 0.2:
        return "I'm sorry, I don't understand. Can you rephrase?"
    return responses[best_match_idx]


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    response = get_response(user_input)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)

# Terminal Version
# def chatbot():
#     print("Chatbot: Hello! Ask me anything or type 'exit' to quit.")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == 'exit':
#             print("Chatbot: Goodbye!")
#             break
#         response = get_response(user_input)
#         print(f"Chatbot: {response}")


# if __name__ == "__main__":
#     chatbot()
