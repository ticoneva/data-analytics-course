# Using LoRA to train a large language model for sequence classification.
# GPT-J is a six-billion parameter model, which is too large to finetune even on a
# single A100 80GB in native precision. We will utilize two techniques to make
# the model trainable on a single GPU:
# 1. Low Rank Adapter (LoRA): instead of fine-tuning the large foundation model,
#    we add the multiplication of two smaller matrices with low rank to it and train 
#    those instead.
# 2. Load foundation model in 8-bit/4-bit: because we are not training the foundation
#    model, we can low it in lower precision without incurring too much loss in 
#    performance.
# Note that 1. is only necessary if you are fine-tuning the model. 2. alone is 
# sufficient for inference.
#
# New HF text classification block is added on top of a pretrained GPT-J model.
# Utilizes early stopping, reloading of best model, learning rate schedule,
# gradient accumulation and multi-GPU support. HF trainer has good default settings 
# so we do not have to provide many settings.
#
# Run on SCRP with A100 GPU
# (Change batch_size to 16 and gradient_accumulation_steps to 2)
# conda activate pytorch
# compute --gpus-per-task=a100 python hf7-lora.py
#
# Run on SCRP with RTX 3090 GPU
# (Change batch_size to 4, max_length to 256 and gradient_accumulation_steps to 8)
# conda activate pytorch
# compute --gpus-per-task=rtx3090 --mem=250G python hf7-lora.py
#
# Change log:
# 2023-10-27 Minor text correction
# 2023-7-17  Initial version

# Settings
model_name = "EleutherAI/gpt-j-6B"      # Pre-trained model to download
max_length = 256                        # Maximum number of tokens per sample
samples = 100                           # Sample size. None means full sample
batch_size = 8                          # Batch size for EACH GPU
gradient_accumulation_steps = 1         # No. of batches to accumulate before updating weights
epochs = 5                              # No. of epochs
cpu_num = 4                             # For batch data processing
seed = 42                               # Seed for data shuffling

# loRA
lora_r = 8                              # Rank of the factorizable LoRA matrix
lora_alpha = 8                          # Learning rate scaling parameter
lora_dropout = 0                        # Dropout rate for LoRA layers

# Storage locations
lora_save_path = "hf7-lora"             # LoRA save path
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
                          EarlyStoppingCallback,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model
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
                                  save_strategy='epoch',
                                  load_best_model_at_end=True,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  gradient_accumulation_steps=gradient_accumulation_steps,
                                  num_train_epochs=epochs,
                                  bf16=True)

# BitsAndBytes configuration
quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                         llm_int8_threshold=6.0)

# Model
# If you load a model not pretrained for the task you specify,
# HF will add an appropriate top-level block for you. In this case,
# you might have to provide some settings, e.g. number of target labels.
# We also provide the quantization configuration here.
model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                           num_labels=2,
                                                           torch_dtype=torch.bfloat16,
                                                           device_map="auto",
                                                           quantization_config=quantization_config)

# This is necessary, otherwise you will get:
# ValueError: Cannot handle batch sizes > 1 if no padding token is defined.
model.config.pad_token_id = model.config.eos_token_id

# LoRA configuration
# Specifying task_type is necessary because it affects the loss function
# and output of the model. For the full list of types, see:
# https://github.com/huggingface/peft/blob/main/src/peft/mapping.py#L47
lora_config = LoraConfig(r=lora_r,
                         lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout,
                         task_type="SEQ_CLS"
                         )

# Add LoRA on top of the foundation model
model = get_peft_model(model, lora_config)

# Print trainable parameter count
model.print_trainable_parameters()

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
#     callbacks=[es_callback],
#     compute_metrics=compute_metrics,

# This starts the actual training
trainer.train()

# Evaluate with test set
trainer.evaluate(dataset["test"])

# Save LoRA
model.save_pretrained(lora_save_path)

### Use model to make prediction on new data ###
# Create HF Dataset from a list and tokenize the data
new_data = [{"text": "This movie is really bad."},
            {"text": "I really like this movie!"}]
new_dataset = Dataset.from_list(new_data)
new_dataset = new_dataset.map(encode, batched=True, num_proc=cpu_num)

# Make predictions. Note that trainer returns numpy array instead of PT tensor
predictions, label_ids, metrics = trainer.predict(new_dataset)

print(predictions[0])
