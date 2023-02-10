# Example fine-tuning a transformer model for the same task on Tensorflow.
# Fine-tuning Hugging Face BERT model with IMDB data.
# Utilizes early stopping, reloading of best model and learning rate schedule.
# Takes ~15min. on a single RTX 3090 to get 88% test accuracy.                 
#
# Run on SCRP with one RTX 3060 GPU:
# conda activate tensorflow
# gpu python hf-3-fine-tune-same-task.py
#
# Run on SCRP with one RTX 3060 GPU:
# conda activate tensorflow
# compute --gpus-per-task=rtx3090 python hf-3-fine-tune-same-task.py
#
# Change log:
# 2022-12-19 Initial version

# Settings
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Pre-trained model to download
samples = None                                                  # Sample size. None means full sample.
epochs = 10                                                     # No. of training epochs
learn_rate = 5e-5                                               # Initial learning rate
cpu_num = 4
batch_size = 64
seed = 42                                                       # Seed for data shuffling

# Storage locations
log_prefix = "logs/transformers/imdb"   # Tensorboard log location
hf_dir = None                           # Cache directory (None means HF default)

import os
import datetime
import tensorflow as tf # Need to import either Tensorflow or PyTorch
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adam
import numpy as np
import time
from transformers import DefaultDataCollator
from datasets import load_dataset,DatasetDict

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

# Set up tokenizer and model. This can be copied from each model's card on Hugging Face
# Tensorflow models starts with 'TF'
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

model.summary()

# Tokenizer wrapper for parallel processing
def encode(examples):
    return tokenizer(examples['text'],
                     truncation=True, 
                     padding='max_length')

# Tokenizes datasets
dataset = dataset.map(encode, batched=True, num_proc=cpu_num)

# Convert HF dataset to TF dataset using model.prepare_tf_dataset
data_collator = DefaultDataCollator(return_tensors="tf")
for split in ["train","valid","test"]:
    dataset[split] =  model.prepare_tf_dataset(
                                                dataset[split],
                                                shuffle=False,
                                                batch_size=batch_size,
                                                collate_fn=data_collator,
                                                )

# Optimizer with learning rate decay
# The number of training steps is the number of samples in the dataset, 
# divided by the batch size then multiplied by the total number of epochs.
num_train_steps = len(dataset["train"]) * epochs
lr_scheduler = PolynomialDecay(
    initial_learning_rate=learn_rate, 
    decay_steps=num_train_steps
)
optimizer = Adam(learning_rate=lr_scheduler)       
    
model.compile(optimizer=optimizer,
              metrics=['accuracy'])

# Tensorboard callback
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_prefix + '-' 
                                             + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 
                                             histogram_freq=1)

# Early stopping callback
es_callback = tf.keras.callbacks.EarlyStopping(patience=3,
                                               restore_best_weights=True)
# Train the model
model.fit(dataset["train"],
          epochs=epochs,
          validation_data=dataset["valid"],
          callbacks=[tb_callback,es_callback])

# Validation
score, acc = model.evaluate(dataset["test"])
print('Test score:', score)
print('Test accuracy:', acc)
