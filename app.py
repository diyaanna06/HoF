from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import os
import pickle
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Set environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Flask application
app = Flask(__name__)

# Configure session to use filesystem (server-side session management)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SESSION_PERMANENT'] = False
Session(app)

# Load Machine Learning Models
model = pickle.load(open("RFmodel.pkl", "rb"))
ferti = pickle.load(open("Fertiliser.pkl", "rb"))

# Routes
@app.route('/')
def index():
    # Initialize chat history in session
    if 'history' not in session:
        session['history'] = []
        welcome_message = "AgriBot: Hello! I am your agricultural assistant. How can I assist you with farming today?"
        session['history'].append({'message': welcome_message, 'sender': 'bot'})
        session.modified = True

    return render_template('index.html', history=session['history'])

@app.route('/chat')
def chat():
    return render_template('index.html')

@app.route('/edit')
def edit_page():
    return render_template('edit.html')

@app.route('/Model1')
def Model1():
    return render_template('Model1.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form inputs
    temp = request.form.get('temp')
    humi = request.form.get('humid')
    mois = request.form.get('mois')
    soil = request.form.get('soil')
    crop = request.form.get('crop')
    nitro = request.form.get('nitro')
    pota = request.form.get('pota')
    phosp = request.form.get('phos')

    # Validate inputs
    if None in (temp, humi, mois, soil, crop, nitro, pota, phosp) or not all(val.isdigit() for val in (temp, humi, mois, soil, crop, nitro, pota, phosp)):
        return render_template('Model1.html', x='Invalid input. Please provide numeric values for all fields.')

    # Convert values to integers
    input_values = [int(temp), int(humi), int(mois), int(soil), int(crop), int(nitro), int(pota), int(phosp)]
    
    # Make prediction
    res = ferti.classes_[model.predict([input_values])]
    
    return render_template('Model1.html', x=res)

@app.route('/submit', methods=['POST'])
def on_submit():
    query = request.form.get('query', '').strip()
    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400

    session['history'].append({'message': query, 'sender': 'user'})
    
    response = generate_response(query)
    response_message = f"AgriBot Response: {response}"
    session['history'].append({'message': response_message, 'sender': 'bot'})
    session.modified = True

    return jsonify({'query': query, 'response': response_message})

def generate_response(query):
    try:
        qa_prompt = (
            "You are an intelligent agriculture assistant providing accurate and actionable advice to farmers. "
            "Your primary role is to guide users on various aspects of farming, including crop selection, climate conditions, "
            "soil health, pest control, irrigation, and sustainable techniques. Your responses should be simple and clear, "
            "tailored to users of different expertise levels."
        )
        input_text = f"{qa_prompt}\nUser question:\n{query}"
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        result = llm.invoke(input_text)

        return result.content if result else "I'm sorry, but I couldn't process your request at the moment. Please try again later."
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
