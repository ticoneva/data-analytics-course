# Using Deepspeed to train a large language model. Note that you should try LoRA
# in hf7-lora.py first before trying Deepspeed.
# GPT-J is a six-billion parameter model, which is too large to train even on a
# single A100 80GB. DeepSpeed allows the training of such model by:
# 1. splitting some components across multiple GPUs. 
# 2. offloading these components to main memory or SSD. 
# The first is obviously only useful when training with multiple GPUs, while 
# the second is useful even with one GPU.
# DeepSpeed has three stages:
# - ZeRO stage 1 splits optimizer states
# - ZeRO stage 2 splits gradient
# - ZeRO stage 3 splits model parameters
# Each stage can be combined with ZeRO-Offload to offload data to main memory/SSD.
# Offloading is CPU intensive, so allocate CPU cores accordingly. 
# New HF text classification block is added on top of a pretrained GPT-J model.
# Utilizes early stopping, reloading of best model, learning rate schedule,
# gradient accumulation and multi-GPU support. HF trainer has good default settings 
# so we do not have to provide many settings.
# 
# DeepSpeed memory requirement estimate:
# https://deepspeed.readthedocs.io/en/latest/memory.html
#
# Run on SCRP with A100 GPU and DeepSpeed ZeRO stage 2
# (Set ds_config = "hf6-ds2.json")
# (Change batch_size to 24 and gradient_accumulation_steps to 1)
# conda activate pytorch
# compute --gpus-per-task=a100 deepspeed hf6-deepspeed.py
#
# Run on SCRP with two RTX 3090 GPU and DeepSpeed ZeRO stage 2
# (Set ds_config = "hf6-ds2.json")
# (Change batch_size to 4, max_length to 256 and gradient_accumulation_steps to 3)
# conda activate pytorch
# compute --gpus-per-task=rtx3090:2 --mem=250G deepspeed hf6-deepspeed.py
#
# Change log:
# 2023-7-16 Switch to dynamic padding
# 2023-3-8  Updated introduction
# 2023-3-3  Initial version

# Settings
model_name = "EleutherAI/gpt-j-6B"      # Pre-trained model to download
max_length = 256                        # Maximum number of tokens per sample
samples = 100                           # Sample size. None means full sample
batch_size = 4                          # Batch size for EACH GPU
gradient_accumulation_steps = 3         # No. of batches to accumulate before updating weights
epochs = 5                              # No. of epochs
cpu_num = 4                             # For batch data processing
seed = 42                               # Seed for data shuffling

# Deepspeed
ds_config = "hf6-ds2.json"            # Deepspeed configuration file

# Storage locations
hf_dir = None                           # Cache directory (None means HF default)
output_dir = "~/large-data/hfpt6"       # Predictions and checkpoints directory
dataset_load_path = None                # Load tokenized dataset. Generate it if none.
dataset_save_path = None                # Save a copy of the tokenized dataset

# Imports
import os
import datetime
import numpy as np
import torch
from transformers import (AutoTokenizer,
                          DataCollatorWithPadding,
                          AutoModelForSequenceClassification,
                          AutoConfig,
                          TrainingArguments,
                          Trainer,
                          EarlyStoppingCallback)
from datasets import load_dataset,Dataset,DatasetDict,load_from_disk
import evaluate


# Set Hugging Face directory
if hf_dir is not None:
    os.environ["HF_HOME"] = hf_dir
    
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# GPT model has no pad token defined
tokenizer.pad_token = tokenizer.eos_token

# Tokenizer wrapper for parallel processing
def encode(examples):
    return tokenizer(examples['text'],
                     truncation=True,
                     padding=False,
                     max_length=max_length
                    )

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

# Set up output dir
output_dir = os.path.expanduser(output_dir)
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
        
# HF class containing hyperparameters
training_args = TrainingArguments(output_dir=output_dir, 
                                  evaluation_strategy="epoch",
                                  save_strategy="epoch",
                                  load_best_model_at_end=True,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  gradient_accumulation_steps=gradient_accumulation_steps,
                                  num_train_epochs=epochs,
                                  bf16=True,
                                  deepspeed=ds_config)

# Model
# If you load a model not pretrained for the task you specify,
# HF will add an appropriate top-level block for you. In this case,
# you might have to provide some settings, e.g. number of target labels.
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# This is necessary, otherwise you will get:
# ValueError: Cannot handle batch sizes > 1 if no padding token is defined.
model.config.pad_token_id = model.config.eos_token_id

# Freeze weights
#for param in model.transformer.parameters():
#    param.requires_grad = False

# Optimizer
# HF Trainer defaults to AdamW with linear learning rate warm up and decay. 
# You only need to specify the optimizer if you want to change its default settings.

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
    callbacks=[es_callback],
    data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
)
#     compute_metrics=compute_metrics,

# This starts the actual training
trainer.train()

# Evaluate with test set
trainer.evaluate(dataset["test"])

### Use model to make prediction on new data ###
# Create HF Dataset from a list and tokenize the data
new_data = [{"text": "This movie is really bad."},
            {"text": "I really like this movie!"}]
new_dataset = Dataset.from_list(new_data)
new_dataset = new_dataset.map(encode, batched=True, num_proc=cpu_num)

# Make predictions. Note that trainer returns numpy array instead of PT tensor
predictions, label_ids, metrics = trainer.predict(new_dataset)

print(predictions[0])
