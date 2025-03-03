# Parallel OpenAI API call example
# Version: 2025-3-3

# Settings
model = "text-small"
api_key="your_api_key"
n_jobs = 2                  # No. of simultaneous calls to model

from multiprocessing import Pool
from openai import OpenAI
import numpy as np
import time

client = OpenAI(
    base_url = 'https://scrp-chat.econ.cuhk.edu.hk/api',
    api_key=api_key,
)

def f(prompt):
    """
    Function to run in each process. Calls OpenAI API.
    """
    response = client.chat.completions.create(
      model=model,  
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
      ],
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    # Load prompts
    prompt_list = ["What is 2 + 3?",
                   "Is a cow bigger or a mouse bigger?"]
	
    # Run 
    with Pool(n_jobs) as p:
        ret_list = p.map(f, prompt_list)

print(ret_list)