# Example fine-tuning a transformer model for a new task on Tensorflow.
# Fine-tuning Hugging Face BERT model with IMDB data.
# New fully-connected layers are added on top of a pretrained BERT model.
# Utilizes early stopping, reloading of best model, learning rate decay and
# multi-GPU support.
#
# Training happens in two stages:
# 1. The fully-connected layers are trained for the specified epochs count.
# 2. The whole model, including the BERT layers, is trained further.
# Takes ~1hr on two RTX 3090 to get 93% test accuracy.
#
# Run on SCRP with one RTX 3060 GPU:
# (Change batch_size to 8)
# conda activate tensorflow
# gpu python hf4-tf-fine-tune-new-task.py
#
# Run on SCRP with two RTX 3090 GPU:
# conda activate tensorflow
# compute --gpus-per-task=rtx3090:2 python hf4-tf-fine-tune-new-task.py
#
# Change log:
# 2022-1-1   Typo correction
# 2022-12-19 Initial version

# Settings
model_name = "bert-base-uncased"        # Pre-trained model to download
samples = None                          # Sample size. None means full sample.
batch_size = 16                         # Batch size for EACH GPU
epochs_1st = 5                          # No. of epochs in the 1st stage
epochs_2nd = 10                         # No. of epochs in the 2nd stage
learn_rate_1st = 0.001                  # Initial learning rate in 1st stage
learn_rate_2nd = 5e-5                   # Initial learning rate in 2nd stage
cpu_num = 4                             # For batch data processing
seed = 42                               # Seed for data shuffling

# Storage locations
log_prefix = "logs/transformers/imdb"   # Tensorboard log location
hf_dir = None                           # Cache directory (None means HF default)
dataset_load_path = None                # Load tokenized dataset. Generate it if none.
dataset_save_path = None                # Save a copy of the tokenized dataset

# Imports
import os
import datetime
from transformers import AutoTokenizer,DefaultDataCollator,TFAutoModel, AutoConfig
# Tensorflow must be imported after tranformers, otherwise will crash
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Dropout
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adam
from datasets import load_dataset,DatasetDict,load_from_disk

# Set Hugging Face directory
if hf_dir is not None:
    os.environ["HF_HOME"] = hf_dir
    
# Create log directory if necessary
log_dir = os.path.dirname(log_prefix)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenizer wrapper for parallel processing
def encode(examples):
    return tokenizer(examples['text'],
                     truncation=True, 
                     padding='max_length')

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
    
    # For tf.data.Dataset it's num_parallel_calls instead of num_proc
    dataset = dataset.map(encode, batched=True, num_proc=cpu_num)

    # HF caches datasets so this is not necessary unless you intend to move the files
    if dataset_save_path is not None:
        dataset.save_to_disk(dataset_save_path)

# Create TF mirrored training strategy
strategy = tf.distribute.MirroredStrategy()
gpu_num = strategy.num_replicas_in_sync
print('Number of devices: {}'.format(gpu_num))    

# Global batch size = individual batch size * GPU count
global_batch_size = batch_size * gpu_num

# Convert HF dataset to TF dataset
# Because we will be adding new layers, model.prepare_tf_dataset does not work
# as it treats target as an input rather than an output
data_collator = DefaultDataCollator(return_tensors="tf")
for split in ["train","valid","test"]:
    dataset[split] = dataset[split].to_tf_dataset(
        columns=["attention_mask", "input_ids", "token_type_ids"],
        label_cols=["labels"],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=global_batch_size,
    )

# Our data is not file based so shard by data
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
for split in ["train","valid","test"]:
    dataset[split] = dataset[split].with_options(options)

with strategy.scope():
    # Model
    config = AutoConfig.from_pretrained(model_name)
    max_length = config.max_position_embeddings

    # Three input layers, corresponding to the three data columns
    input_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="input_ids"
    )
    attention_masks = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="attention_mask"
    )
    token_type_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="token_type_ids"
    )

    # Load the pretrained BERT model and freeze it weights
    bert_model = TFAutoModel.from_pretrained(model_name)
    bert_model.trainable = False
    
    # Feed the output of the BERT model to new fully-connected layers   
    bert_output = bert_model(input_ids, 
                             attention_mask=attention_masks, 
                             token_type_ids=token_type_ids)
    x = Dense(50, activation='relu')(bert_output['pooler_output'])
    output = Dense(1, activation='sigmoid')(x)
    
    # Keras Model
    model = Model(inputs=[input_ids,token_type_ids,attention_masks], 
                  outputs=output)
    print(model.summary())
    
### 1. Train the new fully-connected layers ###    
    
# Optimizer with learning rate decay
# The number of training steps is the number of samples in the dataset, 
# divided by the batch size then multiplied by the total number of epochs.
num_train_steps = len(dataset["train"]) * epochs_1st
lr_scheduler = PolynomialDecay(
    initial_learning_rate=learn_rate_1st, 
    decay_steps=num_train_steps
)
optimizer = Adam(learning_rate=lr_scheduler)    

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Tensorboard callback
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_prefix + '-' 
                                             + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 
                                             histogram_freq=1)

# Early stopping callback
es_callback = tf.keras.callbacks.EarlyStopping(patience=3,
                                               restore_best_weights=True)

# Training the top dense layer
print('Train...')
model.fit(dataset["train"],
          epochs=epochs_1st,
          validation_data=dataset["valid"],
          callbacks=[tb_callback,es_callback])

# Record which epoch we stopped for the next stage
prev_epoch = es_callback.stopped_epoch
if prev_epoch == 0:
    prev_epoch = epochs_1st
print("Training stopped at epoch",prev_epoch)

### 2. Fine-tuning the whole model, including the BERT layer ###
bert_model.trainable = True
print(model.summary())

# Transformer needs smaller initial learning rate
num_train_steps = len(dataset["train"]) * epochs_2nd
lr_scheduler = PolynomialDecay(
    initial_learning_rate=learn_rate_2nd, 
    end_learning_rate=0.0, 
    decay_steps=num_train_steps
)
optimizer = Adam(learning_rate=lr_scheduler)
 
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(dataset["train"],
          epochs=prev_epoch + epochs_2nd,
          initial_epoch=prev_epoch,
          validation_data=dataset["valid"],
          callbacks=[tb_callback,es_callback])

# Validation
score, acc = model.evaluate(dataset["test"])
print('Test score:', score)
print('Test accuracy:', acc)