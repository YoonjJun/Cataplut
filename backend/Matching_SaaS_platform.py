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
               "content": f"You are a    : {problem_1} "}]
)

response_s = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user",
               "content": f"{solution_1} can you be descriptive on this situation 150 words and dont need to provide url"}]
)

problem = response_p.choices[0].message.content
solution = response_s.choices[0].message.content

print(problem)
print(solution)

problem_vector = encode_text_to_vector(problem)
solution_vector = encode_text_to_vector(solution)

# # Example database of company embeddings (in reality, this would be fetched from our actual database)
# company_database = [
#     {"name": "CompanyB", "problem_embedding": ..., "solution_embedding": ...},
#     {"name": "CompanyC", "problem_embedding": ..., "solution_embedding": ...}
#     # Assume embeddings are pre-generated and stored
# ]

# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Function to find matches based on embeddings
# def find_matches(company_a_embeddings, company_database):
#     matches = []
#     for company in company_database:
#         similarity_a_problem_b_solution = cosine_similarity([company_a_embeddings["problem"]], [company["solution_embedding"]])[0][0]
#         similarity_a_solution_b_problem = cosine_similarity([company_a_embeddings["solution"]], [company["problem_embedding"]])[0][0]

#         if similarity_a_problem_b_solution > 0.7 and similarity_a_solution_b_problem > 0.7:  # Example threshold
#             matches.append(company["name"])
#     return matches

# # Find matching companies
# matching_companies = find_matches(company_a_embeddings, company_database)
