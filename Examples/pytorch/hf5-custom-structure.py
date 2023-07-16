# Example fine-tuning a transformer model for a new task on PyTorch, using
# a custom top layer structure. 
# We will fine-tuning Hugging Face BERT model with IMDB data, but have four
# targets instead of one.
# New HF text classification block is added on top of a pretrained BERT model.
# Utilizes early stopping, reloading of best model, learning rate schedule and
# multi-GPU support. HF trainer has good default settings so we do not have to 
# provide many settings.
# 
#
# Run on SCRP with one RTX 3060 GPU:
# (Change batch_size to 8)
# conda activate pytorch
# gpu python hf-pt-5-custom-structure.py
#
# Run on SCRP with two RTX 3090 GPU:
# conda activate pytorch
# compute --gpus-per-task=rtx3090:2 python hf5-custom-structure.py
#
# Change log:
# 2023-7-16 Switch to dynamic padding
# 2023-2-10 Initial version

# Settings
model_name = "bert-base-uncased"        # Pre-trained model to download
samples = 100                           # Sample size. None means full sample.
batch_size = 32                         # Batch size for EACH GPU
epochs = 5                              # No. of epochs
cpu_num = 4                             # For batch data processing
seed = 42                               # Seed for data shuffling

# Storage locations
hf_dir = None                           # Cache directory (None means HF default)
dataset_load_path = None                # Load tokenized dataset. Generate it if none.
dataset_save_path = None                # Save a copy of the tokenized dataset

# Imports
import os
import datetime
import numpy as np
import torch
from transformers import (AutoTokenizer,
                          DataCollatorWithPadding,
                          TrainingArguments,
                          Trainer,
                          EarlyStoppingCallback)
from datasets import load_dataset,Dataset,DatasetDict,load_from_disk
import evaluate
from hf5 import BertFourTargets       # This is our custom model


# Set Hugging Face directory
if hf_dir is not None:
    os.environ["HF_HOME"] = hf_dir
    
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenizer wrapper for parallel processing
def encode(examples):
    inputs = tokenizer(examples['text'],
                     truncation=True, 
                     padding=False)
    # Duplicate the target four times for demo purpose
    try:
        inputs['label'] = [[examples['label'][i],
                            examples['label'][i],
                            examples['label'][i],
                            examples['label'][i]]
                            for i in range(len(examples['label']))]
    except:
        pass
    return inputs

if dataset_load_path:
    # Load tokenized dataset
    dataset = load_from_disk(dataset_load_path)
else:
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
    
    dataset = dataset.map(encode, batched=True, num_proc=cpu_num)

    # HF caches datasets so this is not necessary unless you intend to move the files
    if dataset_save_path is not None:
        dataset.save_to_disk(dataset_save_path)

# HF class containing hyperparameters
training_args = TrainingArguments(output_dir="test_trainer", 
                                  evaluation_strategy="epoch",
                                  save_strategy="epoch",
                                  load_best_model_at_end=True,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  num_train_epochs=epochs)

# Model defined in 
model = BertFourTargets.from_pretrained(model_name)

# Optimizer
# HF Trainer defaults to AdamW with linear learning rate warm up and decay. 
# You only need to specify the optimizer if you want to change its default settings.

# Function for computing evaluation metric
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.where(logits>0, 1, 0)
    return metric.compute(predictions=predictions.flatten(), references=labels.flatten())

# Early stopping callback
es_callback = EarlyStoppingCallback(early_stopping_patience=3)

# HF Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    compute_metrics=compute_metrics,
    callbacks=[es_callback],
    data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
)

# This starts the actual training
trainer.train()

# Evaluate with test set
trainer.evaluate(dataset["test"])

# Save model
#trainer.save_model("path_to_save")

# Load model
#model = BertFourTargets.from_pretrained("path_to_save")

### Use model to make prediction on new data ###
# Create HF Dataset from a list and tokenize the data
new_data = [{"text": "This movie is really bad."},
            {"text": "I really like this movie!"}]
new_dataset = Dataset.from_list(new_data)
new_dataset = new_dataset.map(encode, batched=True, num_proc=cpu_num)

# Make predictions. Note that trainer returns numpy array instead of PT tensor
logits, label_ids, metrics = trainer.predict(new_dataset)

# Our custom model used BCE loss. Use np.where to get the predicted labels.
predictions = np.where(logits>0,1,0)
print(logits[0:2])
print(predictions[0:2])
