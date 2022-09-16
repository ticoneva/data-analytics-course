import os
os.environ["HF_HOME"] = "/data/huggingface/"

from transformers import TFDistilBertForQuestionAnswering
from transformers import DistilBertTokenizerFast
import tensorflow as tf # Need to import either Tensorflow or PyTorch
import numpy as np
import time

# Set up model. Tensorflow models starts with 'TF'
model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

#Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

def batch_inference(question,text):
    inputs = tokenizer(question, text, 
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

question = ["Where is CUHK?" for i in range(1000)]
text = ["CUHK is a university in Hong Kong." for i in range(1000)]

start_t = time.time()
batch_inference(question,text)
end_t = time.time()

print(end_t - start_t)