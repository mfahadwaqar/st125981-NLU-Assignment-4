"""
Flask Alternative for Web Application
Student ID: st125981
Assignment 4 - Task 4

This is an alternative Flask-based implementation if you prefer Flask over Gradio.
To use this:
1. Uncomment flask in requirements.txt
2. Run: python app_flask.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flask import Flask, render_template, request, jsonify

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model classes (same as app.py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configuration and vocabulary
config_path = '../bert_config.json'
vocab_path = '../vocab.json'

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    vocab_size = config['vocab_size']
    n_layers = config['n_layers']
    n_heads = config['n_heads']
    d_model = config['d_model']
    d_ff = config['d_ff']
    d_k = config['d_k']
    d_v = config['d_v']
    n_segments = config['n_segments']
    max_len = config['max_len']
else:
    vocab_size = 50000
    n_layers = 6
    n_heads = 8
    d_model = 768
    d_ff = 768 * 4
    d_k = d_v = 64
    n_segments = 2
    max_len = 256

if os.path.exists(vocab_path):
    with open(vocab_path, 'r') as f:
        word2id = json.load(f)
    vocab_size = len(word2id)
else:
    word2id = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}

# Copy all model classes from app.py
# (For brevity, assuming they are imported or defined)
# ... [Include all the model class definitions here] ...

# Flask app
app = Flask(__name__)

# Global variables for models
bert_encoder = None
classifier_head = None
models_loaded = False


def load_models():
    """Load the trained models"""
    global bert_encoder, classifier_head, models_loaded
    
    # Initialize models
    bert_encoder = BERTEncoder().to(device)
    classifier_head = nn.Linear(d_model * 3, 3).to(device)
    
    # Load weights
    encoder_path = '../sbert_encoder.pth'
    classifier_path = '../sbert_classifier.pth'
    
    try:
        if os.path.exists(encoder_path) and os.path.exists(classifier_path):
            bert_encoder.load_state_dict(torch.load(encoder_path, map_location=device))
            classifier_head.load_state_dict(torch.load(classifier_path, map_location=device))
            bert_encoder.eval()
            classifier_head.eval()
            models_loaded = True
            print("✓ Models loaded successfully!")
        else:
            print("WARNING: Model files not found.")
            models_loaded = False
    except Exception as e:
        print(f"ERROR loading models: {e}")
        models_loaded = False


# Load models on startup
load_models()


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if not models_loaded:
        return jsonify({
            'error': 'Models not loaded. Please train the models first.'
        }), 500
    
    try:
        data = request.json
        premise = data.get('premise', '')
        hypothesis = data.get('hypothesis', '')
        
        if not premise or not hypothesis:
            return jsonify({
                'error': 'Please provide both premise and hypothesis.'
            }), 400
        
        # Tokenize and predict
        premise_ids, premise_mask = tokenize_text(premise)
        hypothesis_ids, hypothesis_mask = tokenize_text(hypothesis)
        
        premise_ids = premise_ids.to(device)
        premise_mask = premise_mask.to(device)
        hypothesis_ids = hypothesis_ids.to(device)
        hypothesis_mask = hypothesis_mask.to(device)
        
        with torch.no_grad():
            u_embeddings = bert_encoder(premise_ids, premise_mask)
            v_embeddings = bert_encoder(hypothesis_ids, hypothesis_mask)
            
            u = mean_pool(u_embeddings, premise_mask)
            v = mean_pool(v_embeddings, hypothesis_mask)
            
            x = configurations(u, v)
            logits = classifier_head(x)
            probabilities = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            prediction = torch.argmax(logits, dim=1).item()
        
        label_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
        
        return jsonify({
            'prediction': label_map[prediction],
            'entailment': float(probabilities[0]),
            'neutral': float(probabilities[1]),
            'contradiction': float(probabilities[2])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting NLI Flask Application...")
    print("Open your browser at: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)
