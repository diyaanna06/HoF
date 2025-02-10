from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Set environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Flask application
app = Flask(__name__)

# Configure session to use filesystem (server-side session management)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'your_secret_key_here'
Session(app)

@app.route('/')
def index():
    # Initialize chat history in session
    if 'history' not in session:
        session['history'] = []
        welcome_message = "AgriBot: Hello! I am your agricultural assistant. How can I assist you with farming today?"
        session['history'].append({'message': welcome_message, 'sender': 'bot'})
    return render_template('index.html', history=session['history'])

@app.route('/submit', methods=['POST'])
def on_submit():
    query = request.form['query']
    session.setdefault('history', []).append({'message': query, 'sender': 'user'})
    
    response = generate_response(query)
    response_message = f"AgriBot Response: {response}"
    session['history'].append({'message': response_message, 'sender': 'bot'})
    
    return jsonify({'query': query, 'response': response_message})

def generate_response(query):
    # Directly generate a response to the user's query using Google Generative AI
    qa_prompt = "You are an intelligent agriculture assistant designed to provide accurate and actionable advice to farmers and agriculture enthusiasts. Your primary role is to guide users on various aspects of farming, including crop selection, climate conditions, soil health, pest control, irrigation practices, and sustainable agricultural techniques. Additionally, you offer insights on weather forecasts, market trends, and best practices to help users optimize their farming operations. Your responses should be clear and simple, tailored to users with different levels of expertise, and provide localized advice when possible by considering the user’s region and climate. Always ensure that your answers are based on the latest agricultural knowledge and research, and offer follow-up recommendations to support users in making well-informed decisions. Maintain a friendly, patient, and supportive tone, and if you are unsure of an answer, provide general guidance or suggest additional resources where the user can find more information."
    input_text = f"{qa_prompt}\nUser question:\n{query}"
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    result = llm.invoke(input_text)
    return result.content

if __name__ == '__main__':
    app.run(debug=True)
