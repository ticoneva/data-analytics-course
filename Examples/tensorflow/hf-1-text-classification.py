# Example using a Hugging Face model for inference in text classification.
# Demonstrates how using underlying model directly is a magnitude faster 
# than using pipeline. 
# Note that significant time is spent loading the model:
# Initializing Tensorflow: ~3s
# Loading model: ~5s
# Inference 1000 samples: ~0.3s
# Uses Tensorflow backend.
#
# Run on SCRP with one RTX 3060 GPU:
# conda activate tensorflow
# gpu python hf-1-text-classification.py
#
# Change log:
# 2022-12-26 Initial version

# Settings
model_name = "finiteautomata/bertweet-base-sentiment-analysis"

import tensorflow as tf # Need to import either Tensorflow or PyTorch
import numpy as np
import time

# This is your data
text = ["This wine is really good." for i in range(1000)]

### 1. Pipeline with GPU ###
# This should take 14-17s on RTX 3060.

start_t = time.time()

# Set device>=0 to use GPU of that ID
from transformers import pipeline
classifier = pipeline('text-classification', 
                      model=model_name,
                      device=0)
classifier(text)
end_t = time.time()
print("Time elapsed with pipeline:",round(end_t - start_t,2))

### 2. Use underlying model ###
# This should take 5-8s on RTX 3060.

# Timer
start_t = time.time()

# Set up tokenizer and model. This can be copied from each model's card on Hugging Face
# Tensorflow models starts with 'TF'
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

def batch_inference(text):
    """
    Inference function , similar to HF pipeline.
    """
    inputs = tokenizer(text, 
                       return_tensors='tf', 
                       truncation=True, 
                       padding=True)

    # Feed data through the model
    return model(inputs).logits

# Run the inference
# HF classification models output logits, not probabilities
# Convert to probability with softmax function
logits = batch_inference(text)
print("Predicted prob.:",tf.nn.softmax(logits[0]))

# Converting to probability is not necessary if you just want the predicted class
predicted_class_id = tf.math.argmax(logits, axis=-1).numpy().astype(int)
print("Predicted class:",predicted_class_id[0])

end_t = time.time()
print("Time elapsed with custom batch inference:",round(end_t - start_t,2))

