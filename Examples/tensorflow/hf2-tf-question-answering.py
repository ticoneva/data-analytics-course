# Example using a Hugging Face model for inference in question-answering.
#
# Run on SCRP with one RTX 3060 GPU:
# conda activate tensorflow
# gpu python hf2-tf-question-answering.py
#
# Change log:
# 2023-12-1  Import tensorflow after transformers
# 2022-12-26 Initial version

# Settings
model_name = "distilbert-base-cased-distilled-squad"
device = "cuda" # 'cpu' or 'cuda' for GPU

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

import numpy as np
import time

### 1. Pipeline with GPU ###
# This should take 15-17s on RTX 3060.

start_t = time.time()

# Set device>=0 to use GPU of that ID
# Tensorflow must be imported after tranformers, otherwise will crash
from transformers import pipeline
import tensorflow as tf 
model = pipeline('question-answering', 
                  model=model_name,
                  device=device)
model(data_dict_list)

end_t = time.time()
print("Time elapsed with pipeline:",round(end_t - start_t,2))

### 2. Use underlying model ###
# This should take 5-6s on RTX 3060.

# Timer
start_t = time.time()

# Set up tokenizer and model. This can be copied from each model's card on Hugging Face
# Tensorflow models starts with 'TF'
from transformers import TFDistilBertForQuestionAnswering,DistilBertTokenizerFast
model = TFDistilBertForQuestionAnswering.from_pretrained(model_name)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

def batch_inference(question,context):
    """
    Inference function , similar to HF pipeline.
    """    
    inputs = tokenizer(question, context, 
                       return_tensors='tf', 
                       truncation=True, 
                       padding=True)

    # Feed data through the model
    outputs = model(inputs)

    # Q&A model outputs the two logit scores for each word.
    # One for its chance of being the start of the answer
    # and one for its chance of being the end
    start_logits = outputs.start_logits.numpy()
    end_logits = outputs.end_logits.numpy()

    # Find the words with the highest score
    start = np.argmax(start_logits, 1)
    end = np.argmax(end_logits, 1)

    # Return the answers
    tokens = [tokenizer.convert_ids_to_tokens(x) for x in inputs["input_ids"].numpy()]
    return [tokenizer.convert_tokens_to_string(x[start[i]:end[i]+1]) for i,x in enumerate(tokens)]

# Run the inference
answers = batch_inference(question,context)
print("Answer:",answers[0])

end_t = time.time()
print("Time elapsed with custom batch inference:",round(end_t - start_t,2))
