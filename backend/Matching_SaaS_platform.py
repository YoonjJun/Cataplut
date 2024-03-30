from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer

def extract_problem_solution(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(-1)
    # Assuming binary classification: 0 for problem, 1 for solution
    # Here you'd implement logic based on your specific fine-tuning task to extract the summarized text
    return problem_summary, solution_summary

# Load the Sentence Transformer model
model_st = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(text):
    return model_st.encode(text)


# New company input (in reality, this would be input by a user from our website)
company_a_input = "Our company struggles with maintaining an efficient inventory system due to outdated software. Meanwhile, we offer an advanced project management tool designed for remote teams."

# Extract and summarize problem and solution
problem_summary, solution_summary = extract_problem_solution(company_a_input, model, tokenizer)

# Generate embeddings
company_a_embeddings = {
    "problem": generate_embeddings(problem_summary),
    "solution": generate_embeddings(solution_summary)
}

# Example database of company embeddings (in reality, this would be fetched from our actual database)
company_database = [
    {"name": "CompanyB", "problem_embedding": ..., "solution_embedding": ...},
    {"name": "CompanyC", "problem_embedding": ..., "solution_embedding": ...}
    # Assume embeddings are pre-generated and stored
]

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to find matches based on embeddings
def find_matches(company_a_embeddings, company_database):
    matches = []
    for company in company_database:
        similarity_a_problem_b_solution = cosine_similarity([company_a_embeddings["problem"]], [company["solution_embedding"]])[0][0]
        similarity_a_solution_b_problem = cosine_similarity([company_a_embeddings["solution"]], [company["problem_embedding"]])[0][0]
        
        if similarity_a_problem_b_solution > 0.7 and similarity_a_solution_b_problem > 0.7:  # Example threshold
            matches.append(company["name"])
    return matches

# Find matching companies
matching_companies = find_matches(company_a_embeddings, company_database)

