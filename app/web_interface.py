from flask import Flask, render_template, request, jsonify
from app.neural_engine import NeuralEngine

app = Flask(__name__)
engine = NeuralEngine()  # Ваш движок

@app.route('/')
def chat():
    return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question', '')
    response = engine.generate_response(question)
    return jsonify({"answer": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)