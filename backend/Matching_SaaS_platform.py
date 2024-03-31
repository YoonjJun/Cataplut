from transformers import BertTokenizer, BertModel
import torch
from g4f.client import Client

client = Client()


# Function to encode text to a fixed vector with BERT
def encode_text_to_vector(text):
    # Load pre-trained model tokenizer (vocabulary) and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

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
problem_1 = "Our company struggles with maintaining an efficient inventory system due to outdated software."
solution_1 = "Meanwhile, we offer an advanced project management tool designed for remote teams."

response_p = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user",
               "content": f"Read the following company description carefully. Identify and summarize the main problems the company is facing. Focus on retaining critical details that accurately represent the challenges while omitting any unnecessary information. Provide the summary in a concise format.\n{problem_1}\nSummarize the problems in no more than 150 words."}]
)

response_s = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user",
               "content": f"Read the following company description carefully. Identify and summarize the main solutions or products the company offers that address these or other problems. Ensure to capture essential details that accurately represent these solutions, avoiding unnecessary information. Provide the summary in a concise format.\n{solution_1}\nSummarize the solutions in no more than 150 words."}]
)

problem = response_p.choices[0].message.content
solution = response_s.choices[0].message.content

print(problem)
print(solution)

problem_vector = encode_text_to_vector(problem)
solution_vector = encode_text_to_vector(solution)

current_company_embeddings = {"name": current_company_name,
                              "problem_embedding": problem_vector,
                              "solution_embedding": solution_vector
                             }

# implement how to store the current_company_embeddings to the company_database
# # Example database of company embeddings (in reality, this would be fetched from our actual database)
# company_database = [
#     {"name": "CompanyB", "problem_embedding": ..., "solution_embedding": ...},
#     {"name": "CompanyC", "problem_embedding": ..., "solution_embedding": ...}
#     # Assume embeddings are pre-generated and stored
# ]

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# # Function to find matches based on embeddings
def find_matches(current_company_embeddings, company_database):
    matches = []
    current_company_name = current_company_embeddings["name"]  # Extract the name of the current company

    for company in company_database:
        if company["name"] == current_company_name:
            continue  # Skip comparing the company with itself

        similarity_a_problem_b_solution = cosine_similarity([current_company_embeddings["problem_embedding"]], [company["solution_embedding"]])[0][0]
        similarity_a_solution_b_problem = cosine_similarity([current_company_embeddings["solution_embedding"]], [company["problem_embedding"]])[0][0]

        if similarity_a_problem_b_solution > 0.7 and similarity_a_solution_b_problem > 0.7:
            matches.append(company["name"])
    
    return matches

# # Find matching companies
# matching_companies = find_matches(company_a_embeddings, company_database)
