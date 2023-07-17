# Run a large-language model in 8-bit precision.
# State-of-the-art large-language models (LLMs) are very powerful, but they 
# often contain too many parameters to load into a single GPU in their native
# 32-bit precision. Loading the model in 8-bit or even 4-bit will cut the the
# memory requirement proportionally.
# Based on https://github.com/tloen/alpaca-lora
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
import torch
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

# Settings
LOAD_8BIT = True
BASE_MODEL = "decapoda-research/llama-7b-hf"
LORA_WEIGHTS = "tloen/alpaca-lora-7b"

# Is GPU available?
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Tokenizer
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL,
                                           force_download=True)

# Load foundation model
model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=LOAD_8BIT,
        torch_dtype=torch.float16,
        device_map="auto",
        )

# Load Alpaca LoRA add-on
model = PeftModel.from_pretrained(
    model,
    LORA_WEIGHTS,
    torch_dtype=torch.float16,
)

# Compile the model to speed up inference. Only works in PyTorch 2 or above
model = torch.compile(model)

def generate_prompt(instruction):
# This function formats the instruction the user provide into the  prompt format 
# the model expects
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

def evaluate(
    instruction,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
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
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
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
    output = tokenizer.decode(s)
    
    # Keep only the part after "### Response:"
    return output.split("### Response:")[1].strip()    

# This is how you use the model
print(evaluate("""
Can you explain what a large language model is?
"""))    