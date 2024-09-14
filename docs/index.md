# Healthy Chat Bot using LLMs

-Author: Daniel Marin
-Institution: MMIA - USFQ

## Index
- [Healthy Chat Bot using LLMs](#healthy-chat-bot-using-llms)
  - [Index](#index)
  - [Introduction](#introduction)
  - [Objective](#objective)
  - [Key Features](#key-features)
  - [General Methodology](#general-methodology)
    - [Data Extraction](#data-extraction)
    - [Data Preprocessing](#data-Preprocessing)
    - [Model Fine-Tuning](#model-fine-tuning)
    - [Chatbot Development](#chatbot-development)
  - [Conclusion](#conclusion)

## Introduction

This project focuses on the development of an intelligent chatbot using Large Scale Language Models (LLMs) to provide information and advice Personalized health and wellness tips.

## Objective

Develop a chatbot that provides personalized information and advice related to tips, recipes, and vitamins based on each user's individual goals.

## Key Features

1. **Personalized Tips**: Offer tailored advice on how to effectively use supplements and vitamins according to the user's specific health and wellness goals, such as improving energy levels, boosting immunity, improving muscle growth, or maintaining overall well-being.

2. **Recipes**: Provide users with healthy recipes that incorporate specific vitamins and supplements, suited to their dietary preferences and health goals.

3. **Vitamin Recommendations**: Suggest appropriate vitamins and supplements based on the user's goals, dietary restrictions, or lifestyle needs. For example, recommending Vitamin D for someone looking to improve bone health or Vitamin C to boost immunity.

## General Methodology

The general methodology of the project is divided into four main phases:

1. Data Extraction
2. Data Preprocessing
3. Model Fine-Tuning
4. Chatbot Development

### Data Extraction

1. **Identify relevant subreddits**: Start by identifying subreddits that discuss supplements, vitamins, recipes, and health tips.

2. **Access the Reddit API**: Use the Reddit API to collect data. You will need to register an application with Reddit to obtain credentials (client ID, client secret, and user agent).
    [Accede a la página de aplicaciones de Reddit.](https://www.reddit.com/prefs/apps "Accede a la página de aplicaciones de Reddit.")

    ![API.png](https://i.postimg.cc/J0ghJ01k/API.png)

1. **Data Extraction**: Write scripts to extract posts and comments relevant to the topics of interest. Use keywords like “tips,” “recipes,” “vitamins,” “supplements,” “energy,” “immunity,” etc., to filter the content.
   
    ```python
        import praw

        # Inicializar el cliente de la API de Reddit
        reddit = praw.Reddit(
            client_id="o90-FNI49wMSLXZSFMy***",
            client_secret="umR06cv2vyynzGFJduk5NEGKWbu***",
            user_agent="SuplementosBot/0.1 by Adept-Depth-7***"
        )

        # Definir los subreddits y palabras clave relevantes
        subreddits = ['supplements', 'vitamins', 'nutrition']
        keywords = ['vitamin', 'supplement', 'nutrient', 'mineral']
    ```

    ### Data Preprocessing

    1. **Clean Data**: Remove unnecessary text such as URLs, HTML tags, special characters, and emojis. Text is also normalized by converting it to lowercase.
        ```python
       def clean_text(text):
                # Check if the text is a list and convert it to a string
                if isinstance(text, list):
                    text = ' '.join(text)
                
                # Remove HTML tags
                text = re.sub(r'<.*?>', '', text)
                # Remove emojis
                text = re.sub(r'[𐀀-􏿿]', '', text)
                # Remove URLs
                text = re.sub(r'http\S+', '', text)
                # Remove special characters and numbers
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                return text
        ```

    2. **Tokenization**: Split the text into individual words or tokens. This can help understand the frequency and importance of different terms related to supplements and vitamins.

    3. **Remove Stop Words**: Remove common words that do not contribute to understanding the context, such as “and,” “the,” “is,” etc.

    4. **Text Classification**: Classify the data extracted in prompt as user questions and completion of possible answers to train the chatbot

        ```python
            import json
            import os
            from datetime import datetime
            import re
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            import spacy

            # Download the necessary NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)

            # Initialize NLTK components
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()

            # Initialize spaCy for lemmatization and entity extraction
            nlp = spacy.load("en_core_web_sm")

            def clean_text(text):
                # Check if the text is a list and convert it to a string
                if isinstance(text, list):
                    text = ' '.join(text)
                
                # Remove HTML tags
                text = re.sub(r'<.*?>', '', text)
                # Remove emojis
                text = re.sub(r'[𐀀-􏿿]', '', text)
                # Remove URLs
                text = re.sub(r'http\S+', '', text)
                # Remove special characters and numbers
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                return text

        ```

### Model Fine-Tuning

1. **Choose a pre-trained model**: the gpt2 training model is chosen
GPT-2 (Generative Pretrained Transformer 2) is a language model based on the Transformers architecture, which generates text in a coherent manner from a given prompt. The model is pre-trained on a large amount of textual data and can be fine-tuned for specific Natural Language Processing (NLP) tasks. The following libraries are essential to work with GPT-2:
with the following libraries:

    #### Libraries Used
    - **Transformers (Hugging Face)**: Provides tools to load pre-trained models and tokenizers, and fine-tune models for specific tasks.
    - **PyTorch**: Framework for building and training deep learning models, used for customizing and training GPT-2.
    - **Datasets (Hugging Face)**: Helps manage and preprocess large datasets for training and evaluation.


2. **Prepare Training Data**: Use the preprocessed and categorized Reddit data to fine-tune the model.



3. **Train the Model**: Perform fine-tuning of the model using the categorized data.

    ```python
    import json
    import torch
    from torch.utils.data import Dataset
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
    import os
    os.environ["WANDB_DISABLED"] = "true"

    # Configuración del archivo JSON de entrada
    JSON_FILE = 'data.json'
    OUTPUT_DIR = './gpt2-finetuned'
    LOGGING_DIR = './logs'

    # Cargar el archivo JSON
    with open(JSON_FILE, 'r') as file:
        data = json.load(file)

    # Extraer y concatenar el contenido de 'prompt' y 'completion'
    texts = [f"{item['prompt']} {item['completion']}" for item in data]

    # Cargar el tokenizador GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Añadir un token de padding al tokenizador
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tokenizar los textos
    encodings = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=128)

    # Preparar el conjunto de datos
    class TextDataset(Dataset):
        def __init__(self, encodings):
            self.input_ids = encodings['input_ids']
            self.attention_mask = encodings['attention_mask']

            # Usar input_ids como etiquetas para la tarea de modelado de lenguaje
            self.labels = self.input_ids.clone()

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'labels': self.labels[idx]
            }

    dataset = TextDataset(encodings)

    # Cargar el modelo GPT-2
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Configurar los argumentos del entrenamiento
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        num_train_epochs=15,
        logging_dir=LOGGING_DIR,
        logging_steps=10,
        save_steps=10_000,
        save_total_limit=2,
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01
    )

    # Configurar el entrenador
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    # Iniciar el entrenamiento
    trainer.train()

    # Guardar el modelo y el tokenizador
    model.save_pretrained('./results')
    tokenizer.save_pretrained('./results')
    ```

    ### Chatbot Development:
    Deployment: The chatbot is deployed on a user-accessible platform such as a website using Gradio.
    ![chatbot.png](https://i.postimg.cc/hjXWmrh2/chatbot.png)

    ```python
        import gradio as gr
        import torch
        from transformers import GPT2Tokenizer, GPT2LMHeadModel

        # Cargar el tokenizador y el modelo ajustado
        tokenizer = GPT2Tokenizer.from_pretrained('./results')
        model = GPT2LMHeadModel.from_pretrained('./results')
        tokenizer.pad_token = tokenizer.eos_token

        def generar_respuesta(prompt):
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_length=100,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return response

        # Crear la interfaz con Gradio
        iface = gr.Interface(fn=generar_respuesta, inputs="text", outputs="text", title="Chatbot")
        iface.launch(share=True)
    ```
    
[![Chatbot Healthy Bot](https://i.postimg.cc/hjXWmrh2/chatbot.png)](https://9647e1623433d31f97.gradio.live/)

## Conclusion

This project seeks to create an interactive and personalized tool that helps users improve their health and well-being through advice, recipes, and vitamin recommendations tailored to their individual needs. Using advanced natural language processing technologies, the chatbot will be able to provide accurate and relevant information, thus improving the user experience in their quest for a healthier lifestyle.







<iframe src="https://182124d255e5e056ff.gradio.live/" width="800" height="600"></iframe>

