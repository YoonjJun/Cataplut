from transformers import BertTokenizer, BertModel
import torch
from g4f.client import Client
from sklearn.metrics.pairwise import cosine_similarity

client = Client()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to encode text to a fixed vector with BERT
def encode_text_to_vector(text):
    # Load pre-trained model tokenizer (vocabulary) and model
    # Encode text
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Get the embeddings of the last layer
    embeddings = model_output.last_hidden_state

    # Pool the embeddings into a single mean vector
    mean_embedding = embeddings.mean(dim=1)

    return mean_embedding


# New company input (in reality, this would be input by a user from our website)
company_name = "CompanyA"
problems = [] # inputs
solutions = [] # inputs
problem_1 = "Our company struggles with maintaining an efficient inventory system due to outdated software."
solution_1 = "Meanwhile, we offer an advanced project management tool designed for remote teams."

prompts = ["""
            Read the following company description carefully. Identify and summarize the main problems
            the company is facing. Focus on retaining critical details that accurately represent the challenges
            while omitting any unnecessary information. Provide the summary in a concise format.
            """,
            """
            Read the following company description carefully. Identify and summarize the main solutions or products the company offers.
            Ensure to capture essential details that accurately represent these solutions, avoiding unnecessary information. Provide the summary in a concise format.
            """
            ]

def f(text, is_sol):
    response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user",
                           "content": prompts[is_sol] + f"{problem} Summarize the problems in no more than 150 words."}]
            )
    return encode_text_to_vector(response.choices[0].message.content)
