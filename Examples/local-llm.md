# How to Run a Local LLM

The way to use LLM locally depends crucially on what you need from the LLM.
1.	You need conversational AI. This usually happens if you intend to provide instructions to the LLM in plain language.
2.	You need an LLM that goes directly from text data to numeric output. This happens for example if you want a pre-trained sentiment classifier.
3.	You need to fine-tune an LLM to your specific data.

## Scenario 1 - Local Conversational AI

There is usually no need to use a local LLM in this scenario, unless data cannot leave your computer. To use a local conversational LLM, you can install one of the several LLM servers---Ollama, GPT4All, etc. Because you mention that you plan to use it with coding, we will focus on API access through Python. We will use Ollama as an example.
1.	Install Ollama

    Download from:
    https://ollama.com/download/Ollama-darwin.zip
    

2.	Start Ollama in a terminal. Replace “llama3.2” with the model of your preference (https://ollama.com/search).

    ```
    ollama run llama3.2
    ```
    Ollama will proceed to download the model. This could take some time if the model is large. Once the model is downloaded and loaded, you will see `>>> Send a message`. 

3. At this point, you can use the model in Python. First install the OpenAI interface in a terminal:
    
    ```
    pip install openai
    ```

    Then in a python script:

    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url = 'http://localhost:11434/v1',
        api_key='ollama',
    )

    response = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": """Is the following text positive or negative? Output 1 if positive, -1 if negative and 0 for neutral.
            
            This movie is very good.
            """}
        ],
    )

    print(response.choices[0].message.content)

    ```

## Scenario 2 - Task-specific LLM

The reason to use a task-specific LLM is speed---an LLM that has been trained to go straight from text to number is going to run much faster. The reason is that the LLM does not need sophisticated language ability to understand your instruction, so the model can be much smaller. Whereas you usually need a 8-billion paramter model to get good instruction following, a 110-million parameter sentiment model often gives sufficiently good result.

1. In a terminal, install the Hugging Face `transformers` library:

    ```
    pip install transformers
    ```

2. In a Python script, 

    ```python
    classifier = pipeline('text-classification')
    classifier("I am very sad today.")
    ```

    You can specific a model if necessary. Models can be found here:
    https://huggingface.co/models?pipeline_tag=text-classification&sort=trending. Here we use a Chinese text classification model:

    ```python
    classifier = pipeline('text-classification', 
        model="uer/roberta-base-finetuned-chinanews-chinese")
    classifier("香港今天很冷。")
    ```

    If you have multiple observations to go through, simply past a list:

    ```
        classifier(["I am very sad today.",
                    "香港今天很冷。"])
    ```

## Scenario 3 - Fine-tuning an LLM

This is by far the hardest. Please refer to the examples here:

a. PyTorch

   https://github.com/ticoneva/data-analytics-course/tree/master/Examples/pytorch

b. Tensorflow

   https://github.com/ticoneva/data-analytics-course/tree/master/Examples/tensorflow