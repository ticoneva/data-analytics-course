# Example fine-tuning a transformer model for the same task on PyTorch.
# Fine-tuning Hugging Face BERT model with IMDB data.
# Utilizes early stopping, reloading of best model, learning rate schedule and
# multi-GPU support. HF trainer has good default settings so we do not have to 
# provide many settings.
# Takes ~15min. on a single RTX 3090 to get 88% test accuracy.                
#
# Run on SCRP with one RTX 3060 GPU:
# conda activate pytorch
# gpu python hf-pt-3-fine-tune-same-task.py
#
# Run on SCRP with one RTX 3060 GPU:
# conda activate pytorch
# compute --gpus-per-task=rtx3090 python hf-pt-3-fine-tune-same-task.py
#
# Change log:
# 2023-1-2 Initial version

# Settings
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Pre-trained model to download
samples = None                                                  # Sample size. None means full sample.
epochs = 10                                                     # No. of training epochs
cpu_num = 4
batch_size = 64
seed = 42                                                       # Seed for data shuffling

# Storage locations
log_prefix = "logs/transformers/imdb"   # Tensorboard log location
hf_dir = None                           # Cache directory (None means HF default)

import os
import datetime
import numpy as np
import time
import torch
from transformers import (AutoTokenizer,DefaultDataCollator,AutoModelForSequenceClassification,
                          TrainingArguments,Trainer,EarlyStoppingCallback)
from datasets import load_dataset,Dataset,DatasetDict
import evaluate

# Set Hugging Face directory
if hf_dir is not None:
    os.environ["HF_HOME"] = hf_dir
    
# Create log directory if necessary
log_dir = os.path.dirname(log_prefix)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Load source data
dataset = load_dataset("imdb")

if samples is not None:
    # Generate small sample
    for split in ["train","test"]:
        dataset[split] = dataset[split].shuffle(seed=seed).select(range(samples))
        
# Train, valid and test sets
tv_datasets = dataset["train"].train_test_split(test_size=0.1, 
                                                shuffle=True,
                                                seed=seed)
dataset["train"] = tv_datasets["train"]
dataset["valid"] = tv_datasets["test"]
dataset["test"] = dataset["test"]               

# HF class containing hyperparameters
training_args = TrainingArguments(output_dir="test_trainer", 
                                  evaluation_strategy="epoch",
                                  save_strategy="epoch",
                                  load_best_model_at_end=True,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  num_train_epochs=epochs)

# Set up tokenizer and model. This can be copied from each model's card on Hugging Face
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenizer wrapper for parallel processing
def encode(examples):
    return tokenizer(examples['text'],
                     truncation=True, 
                     padding='max_length')

# Tokenizes datasets
dataset = dataset.map(encode, batched=True, num_proc=cpu_num)

# Function for computing evaluation metric
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Early stopping callback
es_callback = EarlyStoppingCallback(early_stopping_patience=3)

# HF Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    compute_metrics=compute_metrics,
    callbacks=[es_callback]
)

# This starts the actual training
trainer.train()

# Evaluate with test set
eval_output = trainer.evaluate(dataset["test"])
print("Out-of-sample performance:")
print(eval_output)

### Use model to make prediction on new data ###
# Create HF Dataset from a list and tokenize the data
new_data = [{"text": "This movie is really bad."},
            {"text": "I really like this movie!"}]
new_dataset = Dataset.from_list(new_data)
new_dataset = new_dataset.map(encode, batched=True, num_proc=cpu_num)

# Make predictions. Note that trainer returns numpy array instead of PT tensor
predictions, label_ids, metrics = trainer.predict(new_dataset)

# HF classification models output logits, not probabilities
# Convert to probability with softmax
prob = np.exp(predictions)/sum(np.exp(predictions))
print("Predicted prob.:",prob)

# Converting to probability is not necessary if you just want the predicted class
# argmax(axis=1) means argmax with each sample
predicted_class_id = predictions.argmax(axis=1)
print("Predicted class:", predicted_class_id)
print("Predicted class label:", [model.config.id2label[id] for id in predicted_class_id])