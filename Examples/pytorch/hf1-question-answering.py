# Example using a Hugging Face model for inference in question-answering using PyTorch
# backend. 
#
# Run on SCRP with one RTX 3060 GPU:
# conda activate pytorch
# gpu python hf-pt-1-question-answering.py
#
# Change log:
# 2023-1-1 Initial version

# Settings
model_name = "distilbert-base-cased-distilled-squad"

import torch         # Need to import either Tensorflow or PyTorch
import numpy as np
import time

# Use GPU if available, otherwise use CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

### This is your data ###
# If you use pipeline, you need a list of dictionaries, with each
# dictionary being one pair of question and context. 
sample_input = {
'question': "Where is CUHK?",
'context': "CUHK is a university in Hong Kong."
}
data_dict_list = [sample_input for i in range(1000)]

# If you use the underlying model, the questions are in one array
# and the context in another. 
question = ["Where is CUHK?" for i in range(1000)]
context = ["CUHK is a university in Hong Kong." for i in range(1000)]

### 1. Pipeline with GPU ###
# This should take ~15s on RTX 3060.

start_t = time.time()

from transformers import pipeline
model = pipeline('question-answering', 
                  model=model_name,
                  device=device)
model(data_dict_list)

end_t = time.time()
print("Time elapsed with pipeline:",round(end_t - start_t,2))

### 2. Use underlying model ###
# Unlike Tensorflow, the PyTorch backend does *not* use GPU during inference by default. 
# To use GPU, we need to manually move the model and the data to GPU.
# This should take ~3s on RTX 3060.

# Timer
start_t = time.time()

# Set up tokenizer and model. This can be copied from each model's card on Hugging Face
from transformers import DistilBertForQuestionAnswering,DistilBertTokenizerFast
model = DistilBertForQuestionAnswering.from_pretrained(model_name)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# Move model to GPU
model = model.to(device)

def batch_inference(question,context):
    """
    Inference function , similar to HF pipeline.
    """    
    inputs = tokenizer(question, context, 
                       return_tensors='pt', 
                       truncation=True, 
                       padding=True)
    
    # Move data to GPU
    inputs = inputs.to(device)
    
    # Feed data through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Q&A model outputs the two logit scores for each word.
    # One for its chance of being the start of the answer
    # and one for its chance of being the end
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    # Find the words with the highest score
    # argmax(dim=1) means argmax with each sample
    start = start_logits.argmax(dim=1)
    end = end_logits.argmax(dim=1)
    
    # Return the answers
    # This is the point where we move the prediction back to main memory with .cpu()
    tokens = [tokenizer.convert_ids_to_tokens(x) for x in inputs["input_ids"].cpu().numpy()]
    return [tokenizer.convert_tokens_to_string(x[start[i]:end[i]+1]) for i,x in enumerate(tokens)]

# Run the inference
answers = batch_inference(question,context)
print("Answer:",answers[0])

end_t = time.time()
print("Time elapsed with custom batch inference:",round(end_t - start_t,2))
