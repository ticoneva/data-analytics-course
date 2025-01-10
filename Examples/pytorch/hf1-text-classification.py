# Example using a Hugging Face model for inference in text classification using 
# PyTorch backend.
# Demonstrates how using underlying model directly can be a magnitude faster 
# than using pipeline:
#   Option 1: Use pipeline with data in a list. ~18s per 1000 samples
#   Option 2: Use pipeline with data in HF Dataset format. ~3s per 1000 samples
#   Option 3: Use underlying model directly. ~2s per 1000 samples
# The speed up in 2 and 3 largely comes from batch-processing the samples on GPU.
# On CPU, batching is not necessarily a good idea. See:
# https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching
#
# See also: https://huggingface.co/docs/transformers/tasks/sequence_classification
#
# Run on SCRP with one RTX 3060 GPU:
# conda activate pytorch
# gpu python hf1-text-classification.py
#
# Change log:
# 2023-11-29 New HF Dataset example
#            Move GPU detection up to top
# 2023-1-1   Initial version

# Settings
model_name = "finiteautomata/bertweet-base-sentiment-analysis"

import torch         # Need to import either Tensorflow or PyTorch
import numpy as np
import time

# Use GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# This is your data
text = ["This wine is really good." for i in range(1000)]

### 1. Pipeline with GPU ###
# This should take ~18s on RTX 3060.

print("Pipeline + list:")

start_t = time.time()

# Load model
from transformers import pipeline
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
# Unlike Tensorflow, the PyTorch backend does *not* use GPU during inference by default. 
# To use GPU, we need to manually move the model and the data to GPU.
# This should take ~2s on RTX 3060.

print("Use underlying model:")

# Timer
start_t = time.time()

# Set up tokenizer and model. This can be copied from each model's card on Hugging Face
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

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
logits = batch_inference(text)

# Convert to probability with softmax
# .cpu() transfers the tensor back to CPU and .numpy() converts its to numpy array
m = torch.nn.Softmax(dim=1)
prob = m(logits).cpu().numpy()
print("Predicted prob.:",prob[0])

# Converting to probability is not necessary if you just want the predicted class
# argmax(dim=1) means argmax with each sample
predicted_class_id = logits.argmax(dim=1).cpu().numpy()

# Print out one example
print("Predicted class:",predicted_class_id[0])
print(model.config.id2label[predicted_class_id[0]])

end_t = time.time()
print("Time elapsed with custom batch inference:",round(end_t - start_t,2))

