# Example using a Hugging Face model for inference in text classification using 
# PyTorch backend.
# Demonstrates how using underlying model directly is a magnitude faster than 
# using pipeline. 
# See also: https://huggingface.co/docs/transformers/tasks/sequence_classification
#
# Run on SCRP with one RTX 3060 GPU:
# conda activate pytorch
# gpu python hf-pt-1-text-classification.py
#
# Change log:
# 2022-1-1 Initial version

# Settings
model_name = "finiteautomata/bertweet-base-sentiment-analysis"

import torch         # Need to import either Tensorflow or PyTorch
import numpy as np
import time

# This is your data
text = ["This wine is really good." for i in range(1000)]

### 1. Pipeline with GPU ###
# This should take ~18s on RTX 3060.

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
# Unlike Tensorflow, the PyTorch backend does *not* use GPU during inference by default. 
# To use GPU, we need to manually move the model and the data to GPU.
# This should take ~3s on RTX 3060.

# Timer
start_t = time.time()

# Set up tokenizer and model. This can be copied from each model's card on Hugging Face
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Use GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Move model to GPU
model = model.to(device)

def batch_inference(text):
    """
    Inference function , similar to HF pipeline.
    """
    inputs = tokenizer(text, 
                       return_tensors="pt",
                       truncation=True, 
                       padding=True)
    
    # Move data to GPU
    inputs = inputs.to(device)
    
    # Feed data through the model
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits

# Run the inference
# HF classification models output logits, not probabilities
# Convert to probability with softmax
# .cpu() transfers the tensor back to CPU and .numpy() converts its to numpy array
logits = batch_inference(text)
m = torch.nn.Softmax(dim=1)
prob = m(logits).cpu().numpy()
print("Predicted prob.:",prob[0])

# Converting to probability is not necessary if you just want the predicted class
# argmax(dim=1) means argmax with each sample
predicted_class_id = logits.argmax(dim=1).cpu().numpy()
print("Predicted class:",predicted_class_id[0])
print(model.config.id2label[predicted_class_id[0]])

end_t = time.time()
print("Time elapsed with custom batch inference:",round(end_t - start_t,2))

