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
# gpu python hf1-tf-text-classification.py
#
# Change log:
# 2023-12-1  Import tensorflow after transformers
# 2022-12-26 Initial version

# Settings
model_name = "finiteautomata/bertweet-base-sentiment-analysis" # Model name
device = "cuda" # 'cpu' or 'cuda' for GPU

import numpy as np
import time

# This is your data
text = ["This wine is really good." for i in range(1000)]

### 1. Pipeline with GPU ###
# This should take ~18s on RTX 3060.

print("Pipeline + list:")

start_t = time.time()

# Load model
# Tensorflow must be imported after tranformers, otherwise will crash
from transformers import pipeline
import tensorflow as tf 
classifier = pipeline('text-classification', 
                      model=model_name,
                      device=device)

# This runs the data through the model. 
# Output is a list, in the same order as the input list
output = classifier(text)

# Print out one example
print(output[0])

end_t = time.time()
print("Time elapsed with pipeline:",round(end_t - start_t,2))

### 2. Pipeline + Datasets with GPU ###
# This should take ~3s on RTX 3060 with batch_size = 16.
batch_size = 16

print("Pipeline + Datasets:")

start_t = time.time()

# Construct a HF Dataset from a list
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
text2 = [{'text':"This wine is really good."} for i in range(1000)]
dataset = Dataset.from_list(text2)

# Load model
from transformers import pipeline
classifier = pipeline('text-classification', 
                      model=model_name,
                      device=device)

# Run the dataset through the model and save results to a list
output_list = []
for out in classifier(KeyDataset(dataset, "text"),batch_size=batch_size):
    output_list.append(out)
    
# Print out one example
print(output_list[0])

end_t = time.time()
print("Time elapsed with pipeline + datasets:",round(end_t - start_t,2))

### 3. Use underlying model ###
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
logits = batch_inference(text)

# Convert to probability with softmax function
# Need to use Tensorflow methods here
print("Predicted prob.:",tf.nn.softmax(logits[0]))

# Converting to probability is not necessary if you just want the predicted class
predicted_class_id = tf.math.argmax(logits, axis=-1).numpy().astype(int)

# Print out one example
print("Predicted class:",predicted_class_id[0])
print(model.config.id2label[predicted_class_id[0]])

end_t = time.time()
print("Time elapsed with custom batch inference:",round(end_t - start_t,2))

