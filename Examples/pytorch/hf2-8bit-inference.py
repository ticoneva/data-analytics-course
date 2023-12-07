# Run a large-language model in 8-bit precision.
# State-of-the-art large-language models (LLMs) are very powerful, but they 
# often contain too many parameters to load into a single GPU in their native
# 32-bit precision. Loading the model in 8-bit or even 4-bit will cut the the
# memory requirement proportionally. A 1 billion-parameter model loaded in 
# 8-bit precision will require approximately 1GB of GPU memory.
#
# Run on SCRP with one RTX 3060 GPU:
# conda activate pytorch
# gpu python hf2-8bit-inference.py
#
# Run on SCRP with one RTX 3090 GPU:
# conda activate pytorch
# compute --gpus-per-task=rtx3090 python hf2-8bit-inference.py
#
# Change log:
# 2023-7-17 Initial version


import sys
import time
import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          AutoConfig, GenerationConfig)

# Settings
LOAD_8BIT = True
model = "mosaicml/mpt-7b-instruct"
tokenizer_model = "EleutherAI/gpt-neox-20b"

### 1. Pipeline with GPU ###
from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
pipe = pipeline('text-generation', model=model, 
                tokenizer=tokenizer, 
                device_map="auto", 
                model_kwargs={"load_in_8bit": True})
with torch.autocast('cuda', dtype=torch.bfloat16):
    output = pipe('Can you explain what a large language model is?',
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.75,
                    top_k=40,
                    use_cache=True)
    print(output[0]['generated_text'])
    
    
### 2. Underlying Model with GPU ###

# Is GPU available?
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    

# Model Configuration
config = AutoConfig.from_pretrained(model, trust_remote_code=True)
config.init_device = 'cuda:0' # For fast initialization directly on GPU!    

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
tokenizer.pad_token = tokenizer.eos_token

# Load foundation model
model = AutoModelForCausalLM.from_pretrained(
        model,
        config=config,
        load_in_8bit=LOAD_8BIT,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
        )

# Compile the model to speed up inference. Only works in PyTorch 2 or above
model = torch.compile(model)

def generate_prompt(instruction):
# This function formats the instruction the user provide into the  prompt format 
# the model expects. Note that this is model specific.
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

def evaluate(
    instruction,
    do_sample=True,
    temperature=0.6,
    top_p=0.75,
    top_k=40,
    num_beams=1,
    max_new_tokens=128,
    **kwargs,
):
# This function takes a text instruction and run it through the model.
# You can optionally provide parameters can affect the length and randomness of
# the model's response.
  
    # Format the prompt
    prompt = generate_prompt(instruction)
    
    # Convert the prompt to tokens
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move the tokens to GPU
    input_ids = inputs["input_ids"].to(device)
    
    # Inference settings
    generation_config = GenerationConfig(
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        pad_token_id=tokenizer.eos_token_id,
        **kwargs,
    )
    
    # torch.no_grad() tells PyTorch there is no need to keep track of gradient
    # because we are not training the model
    with torch.no_grad():
        
        # model.generate() runs the tokenized prompt through the model
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
        
    # Decode the model's output
    s = generation_output.sequences[0]
    output = tokenizer.decode(s,skip_special_tokens=True)
    
    # Keep only the part after "### Response:"
    return output.split("### Response:")[1].strip()    



# The question
question = "Can you explain what a large language model is?"

# Start time
start = time.time()

print(question)

# This is how you use the model
print(evaluate(question))    

print(round(time.time() - start,2),'s')