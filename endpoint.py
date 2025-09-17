import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, TFBertModel

app = FastAPI()

class PredictionRequest(BaseModel):
    text: str
    model_dir: str = "./results"
    threshold: float = 0.5

class BertClassifier(tf.keras.Model):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.out = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        input_ids, attention_mask = inputs
        bert_output = self.bert(input_ids, attention_mask=attention_mask)[0]
        cls_embedding = bert_output[:, 0, :]
        x = self.dense1(cls_embedding)
        x = self.dropout(x, training=training)
        return self.out(x)

def load_model_from_weights(model_dir="./results", max_length=128):
    weights_path = os.path.join(model_dir, "urgency_classifier_weights.h5")
    tokenizer_dir = os.path.join(model_dir, "tokenizer")
    bert_dir = os.path.join(model_dir, "bert_model")

    if not os.path.exists(weights_path):
        raise RuntimeError(f"Weights not found: {weights_path}")

    if os.path.isdir(tokenizer_dir):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if os.path.isdir(bert_dir):
        bert = TFBertModel.from_pretrained(bert_dir)
    else:
        bert = TFBertModel.from_pretrained("bert-base-uncased")

    model = BertClassifier(bert)

    dummy_input_ids = tf.zeros((1, max_length), dtype=tf.int32)
    dummy_attention = tf.zeros((1, max_length), dtype=tf.int32)
    _ = model((dummy_input_ids, dummy_attention), training=False)

    model.load_weights(weights_path)
    return model, tokenizer

MODEL_DIR = "./results"
model, tokenizer = load_model_from_weights(MODEL_DIR)

@app.post("/predict-urgency")
async def predict_urgency(request: PredictionRequest):
    try:
        encoding = tokenizer(
            request.text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )
        input_ids = tf.cast(encoding["input_ids"], tf.int32)
        attention_mask = tf.cast(encoding["attention_mask"], tf.int32)

        prob = float(model((input_ids, attention_mask), training=False).numpy().flatten()[0])
        predicted = "urgency detected" if prob > request.threshold else "no significant urgency detected"

        return {
            "probability": prob,
            "class": predicted,
            "method": "weights_load"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
