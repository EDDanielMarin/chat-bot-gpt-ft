import json
import os
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

# Download the necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Initialize spaCy for lemmatization and entity extraction
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    # Check if the text is a list and convert it to a string
    if isinstance(text, list):
        text = ' '.join(text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove emojis
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def normalize_text(text):
    # Check if the text is a list and convert it to a string
    if isinstance(text, list):
        text = ' '.join(text)
        
    # Tokenization
    tokens = nltk.word_tokenize(text.lower())
    # Remove stopwords and apply lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def process_conversations(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    formatted_data = []
    
    # Iterate over each conversation
    for conversation in data:
        conversation_data = []
        
        # Ensure questions/answers are strings and not lists
        question = conversation.get('question', '')
        if isinstance(question, list):
            question = ' '.join(question)
        
        answers = conversation.get('answers', [])
        if isinstance(answers, list):
            answers = ' '.join(answers)

        # Create "user" and "assistant" exchanges in the format
        user_message = {
            'role': 'user',
            'content': clean_text(question)
        }
        assistant_message = {
            'role': 'assistant',
            'content': clean_text(answers)
        }
        
        conversation_data.append(user_message)
        conversation_data.append(assistant_message)

        formatted_data.append(conversation_data)

    # Save the processed data to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath('data/processed')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'processed_conversations_{timestamp}.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=4)

    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    try:
        raw_data_dir = os.path.abspath('data/raw')
        latest_file = max(os.listdir(raw_data_dir), key=lambda f: os.path.getctime(os.path.join(raw_data_dir, f)))
        input_file_path = os.path.join(raw_data_dir, latest_file)
        process_conversations(input_file_path)
    except Exception as e:
        print(f"An error occurred: {e}")
