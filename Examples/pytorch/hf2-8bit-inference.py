# Example using a Hugging Face model for inference in text classification using 
# PyTorch backend.
# Demonstrates how using underlying model directly is a magnitude faster than 
# using pipeline. 
# See also: https://huggingface.co/docs/transformers/tasks/sequence_classification
#
# Run on SCRP with one RTX 3060 GPU:
# conda activate pytorch
# gpu python hf1-text-classification.py
#
# Change log:
# 2022-1-1 Initial version

# Settings
model_name = "togethercomputer/GPT-NeoXT-Chat-Base-20B"


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          padding_side='left')
quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                         llm_int8_threshold=6.0)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                            torch_dtype=torch.float16,
                                            device_map="auto",
                                            quantization_config=quantization_config
                                            )

# GPT model has no pad token defined
tokenizer.pad_token = tokenizer.eos_token

