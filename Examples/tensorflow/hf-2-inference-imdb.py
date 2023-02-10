# Example using a Hugging Face model for inference in text classification
# using IMDB data in Hugging Face Dataset format. Uses Tensorflow backend.
#
# Run on SCRP with one RTX 3060 GPU:
# conda activate tensorflow
# gpu python hf-2-inference-imdb.py
#
# Change log:
# 2022-12-19 Initial version

# Settings
model_name = "finiteautomata/bertweet-base-sentiment-analysis"  # Pre-trained model to download
samples = None                                                  # Sample size. None means full sample.
cpu_num = 4
batch_size = 64
seed = 42                                                       # Seed for data shuffling

import tensorflow as tf # Need to import either Tensorflow or PyTorch
import numpy as np
import time
from datasets import load_dataset
from transformers import DefaultDataCollator

# Timer
start_t = time.time()

# Load source data
dataset = load_dataset("imdb")

if samples is not None:
    # Generate small sample
    for split in ["train","test"]:
        dataset[split] = dataset[split].shuffle(seed=seed).select(range(samples))

# Set up tokenizer and model. This can be copied from each model's card on Hugging Face
# Tensorflow models starts with 'TF'
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenizer wrapper for parallel processing
def encode(examples):
    return tokenizer(examples['text'],
                     truncation=True, 
                     padding='max_length')

# Tokenizes datasets
dataset = dataset.map(encode, batched=True, num_proc=cpu_num)

# Convert HF dataset to TF dataset
data_collator = DefaultDataCollator(return_tensors="tf")
for split in ["train","test"]:
    dataset[split] = model.prepare_tf_dataset(
                                    dataset[split],
                                    shuffle=False,
                                    batch_size=batch_size,
                                    collate_fn=data_collator,
                                    )

# Run the inference
# HF classification models output logits, not probabilities
# Convert to probability with softmax function
logits = model.predict(dataset["train"]).logits
print("Predicted prob. for 1st sample:",tf.nn.softmax(logits[0]))

# Converting to probability is not necessary if you just want the predicted class
predicted_class_id = tf.math.argmax(logits, axis=-1).numpy().astype(int)
print("Predicted class for 1st sample:",predicted_class_id[0])

# To use Kera's evaluate function, you need to compile the model
model.compile(metrics=['accuracy'])
model.evaluate(dataset["train"])

end_t = time.time()
print("Time elapsed:",round(end_t - start_t,2))

