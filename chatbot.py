import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

# Load FAQ data
with open('faq_data.json') as f:
    faq_data = json.load(f)

questions = [item['question'] for item in faq_data]
answers = [item['answer'] for item in faq_data]

# Vectorize questions
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# Function to get response
def get_response(user_question):
    user_vec = vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vec, question_vectors)
    best_match_idx = similarity.argmax()
    if similarity[0][best_match_idx] > 0.2:
        return answers[best_match_idx]
    else:
        return "Sorry, I didn't understand your question."

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    response = get_response(user_input)
    print("Bot:", response)
