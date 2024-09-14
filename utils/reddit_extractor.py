import praw
import json
import os
from datetime import datetime

# Initialize the Reddit API client
reddit = praw.Reddit(
    client_id="",
    client_secret="",
    user_agent="SuplementosBot/0.1 by Adept-Depth-7720"
)

# Define relevant subreddits and keywords
subreddits = ['supplements', 'vitamins', 'nutrition']
keywords = ['vitamin', 'supplement', 'nutrient', 'mineral']

# Function to check if the post is a question
def is_question(post_title):
    return post_title.strip().endswith('?') or any(keyword in post_title.lower() for keyword in ['how', 'what', 'why', 'when', 'which'])

# Check if the post is relevant
def is_relevant_post(post_title, post_text):
    return any(keyword in post_title.lower() or keyword in post_text.lower() for keyword in keywords)

# Extract relevant posts and their comments (questions and answers)
def extract_reddit_data():
    data = []
    
    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        
        for post in subreddit.hot(limit=100):  # Limit to 100 hot posts
            if is_relevant_post(post.title, post.selftext) and is_question(post.title):
                # Try to get the top comments as answers
                answers = []
                post.comments.replace_more(limit=0)  # To load all comments
                for comment in post.comments.list():
                    if len(answers) < 3:  # Limit to 3 answers
                        answers.append({
                            'author': str(comment.author),
                            'text': comment.body,
                            'score': comment.score
                        })
                
                data.append({
                    'question': post.title,
                    'text': post.selftext,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': post.created_utc,
                    'subreddit': subreddit_name,
                    'answers': answers
                })
    
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'reddit_questions_{timestamp}.json'
    
    # Path where the data will be saved
    save_path = os.path.join('data', 'raw')
    
    try:
        # Ensure the raw data directory exists
        os.makedirs(save_path, exist_ok=True)
        
        # Full file path including directory and filename
        file_path = os.path.join(save_path, filename)
        
        # Save the extracted data to a JSON file
        with open(file_path, 'w') as f:
            json.dump(data, f)
        
        print(f"Extracted {len(data)} relevant questions. Saved to {file_path}")
    
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    extract_reddit_data()
