import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gradio as gr

# Add parent directory to path to import from notebook
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Model Architecture
# Load configuration
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
    # Default values if config not found
    vocab_size = 50000  # Will be updated after loading vocab
    n_layers = 6
    n_heads = 8
    d_model = 768
    d_ff = 768 * 4
    d_k = d_v = 64
    n_segments = 2
    max_len = 256

# Load vocabulary
if os.path.exists(vocab_path):
    with open(vocab_path, 'r') as f:
        word2id = json.load(f)
    id2word = {int(i): w for w, i in word2id.items()}
    vocab_size = len(word2id)
    print(f"Loaded vocabulary: {vocab_size} words")
else:
    print("WARNING: Vocabulary file not found!")
    word2id = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    id2word = {0: '[PAD]', 1: '[CLS]', 2: '[SEP]', 3: '[MASK]'}

# Model components
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.linear(context)
        return self.norm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        output = self.fc2(F.gelu(self.fc1(x)))
        return self.norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class BERTEncoder(nn.Module):
    def __init__(self):
        super(BERTEncoder, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
    
    def forward(self, input_ids, attention_mask):
        segment_ids = torch.zeros_like(input_ids)
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        
        for layer in self.layers:
            output, _ = layer(output, enc_self_attn_mask)
        
        return output


def mean_pool(token_embeds, attention_mask):
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
    return pool


def configurations(u, v):
    uv = torch.sub(u, v)
    uv_abs = torch.abs(uv)
    x = torch.cat([u, v, uv_abs], dim=-1)
    return x


# Load Models
print("Loading models...")

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
        print("Models loaded successfully!")
        models_loaded = True
    else:
        print("WARNING: Model files not found. Please train the models first.")
        print(f"Looking for:")
        print(f"  - {encoder_path}")
        print(f"  - {classifier_path}")
        models_loaded = False
except Exception as e:
    print(f"ERROR loading models: {e}")
    models_loaded = False


# Text Preprocessing
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,!?;:]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_text(text, max_length=128):
    cleaned = clean_text(text)
    tokens = [word2id.get(word, word2id.get('[MASK]', 3)) for word in cleaned.split()]
    
    # Add [CLS] and [SEP]
    tokens = [word2id['[CLS]']] + tokens + [word2id['[SEP]']]
    
    # Truncate or pad
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    
    attention_mask = [1] * len(tokens)
    
    # Pad
    padding_length = max_length - len(tokens)
    tokens += [0] * padding_length
    attention_mask += [0] * padding_length
    
    return torch.LongTensor(tokens).unsqueeze(0), torch.LongTensor(attention_mask).unsqueeze(0)


# Prediction Function
def predict_nli(premise, hypothesis):
    if not models_loaded:
        # Return error as a label with 100% confidence
        return {"Error - Models not loaded": 1.0}
    
    if not premise or not hypothesis:
        # Return error as a label with 100% confidence
        return {"Error - Missing input": 1.0}
    
    try:
        # Tokenize inputs
        premise_ids, premise_mask = tokenize_text(premise)
        hypothesis_ids, hypothesis_mask = tokenize_text(hypothesis)
        
        # Move to device
        premise_ids = premise_ids.to(device)
        premise_mask = premise_mask.to(device)
        hypothesis_ids = hypothesis_ids.to(device)
        hypothesis_mask = hypothesis_mask.to(device)
        
        # Inference
        with torch.no_grad():
            # Encode sentences
            u_embeddings = bert_encoder(premise_ids, premise_mask)
            v_embeddings = bert_encoder(hypothesis_ids, hypothesis_mask)
            
            # Mean pooling
            u = mean_pool(u_embeddings, premise_mask)
            v = mean_pool(v_embeddings, hypothesis_mask)
            
            # Concatenate
            x = configurations(u, v)
            
            # Classify
            logits = classifier_head(x)
            probabilities = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            prediction = torch.argmax(logits, dim=1).item()
        
        # Map prediction to label
        label_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
        predicted_label = label_map[prediction]
        
        # Create result dictionary with float probabilities for Gradio Label component
        result = {
            "Entailment": float(probabilities[0]),
            "Neutral": float(probabilities[1]),
            "Contradiction": float(probabilities[2]),
        }
        
        return result
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"Error - Prediction failed": 1.0}


# Gradio Interface
# Example inputs
examples = [
    ["A man is playing a guitar on stage.", "The man is performing music."],
    ["A woman is cutting vegetables.", "Someone is preparing food."],
    ["A dog is running in the park.", "A cat is sleeping on the couch."],
    ["The sun rises in the east.", "The sun sets in the west."],
    ["A child is reading a book.", "A young person is engaged in an activity."],
]

# Create Gradio interface
interface = gr.Interface(
    fn=predict_nli,
    inputs=[
        gr.Textbox(
            label="Premise",
            placeholder="Enter the premise (first sentence)...",
            lines=3
        ),
        gr.Textbox(
            label="Hypothesis",
            placeholder="Enter the hypothesis (second sentence)...",
            lines=3
        ),
    ],
    outputs=gr.Label(label="Prediction Results", num_top_classes=3),
    title="Natural Language Inference (NLI) Predictor",
    description="""
    This application predicts the logical relationship between two sentences:
    - **Entailment**: The hypothesis logically follows from the premise
    - **Neutral**: The hypothesis is neither entailed nor contradicted by the premise
    - **Contradiction**: The hypothesis contradicts the premise
    """,
    examples=examples
)

# Launch the interface
if __name__ == "__main__":
    print("Starting NLI Web Application...")
    interface.launch(
        share=False,
        server_name="127.0.0.1"
    )
