from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import mysql.connector
from dotenv import load_dotenv
import os

from transformers import BertTokenizer, BertModel
import torch
from g4f.client import Client
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

templates = Jinja2Templates(directory="../templates")

@app.get("/")
def read_root():
    # db.test_db()
    return {"Hello": "World"}

@app.get("/test")
async def test(request: Request):
    return templates.TemplateResponse("home.html", {"request":request})

@app.post("/company")
async def create_company(name: str, prob: str, sol: str):
    db.insert_company(name)
    db.insert_problem(name, prob)
    db.insert_solution(name, sol)
    return {}

load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQL_DB")

mydb = mysql.connector.connect(
  host=MYSQL_HOST,
  user=MYSQL_USER,
  password=MYSQL_PASSWORD,
  database=MYSQL_DB
)

def test_db():
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM test")
    myresult = mycursor.fetchall()
    for x in myresult:
        print(x)

def insert_company(name, prob_text, sol_text):
    mycursor = mydb.cursor()
    sql = "INSERT INTO Company (name) VALUES (%s)"
    val = (name)
    mycursor.execute(sql, val)
    mydb.commit()

    company_id = get_company_id(name)
    prob_emb = ai.get_embedding(prob_text)
    sol_emb = ai.get_embedding(sol_text)

    calc_similarity(company_id, prob_emb, is_sol = 0)
    calc_similarity(company_id, sol_emb, is_sol = 1)
    insert_problem(company_id, prob_text, prob_emb)
    insert_solution(company_id, sol_text, sol_emb)

def insert_problem(company_id, prob_text, prob_emb):
    mycursor = mydb.cursor()
    sql = "INSERT INTO Problem (companyId, probText, probEmb) VALUES (%s, %s, %s)"
    val = (company_id, prob_text, prob_emb)
    mycursor.execute(sql, val)
    mydb.commit()

def insert_solution(company_id, sol_text, sol_emb):
    mycursor = mydb.cursor()
    sql = "INSERT INTO Solution (companyId, solText, solEmb) VALUES (%s, %s, %s)"
    val = (company_id, sol)
    mycursor.execute(sql, val)
    mydb.commit()

def calc_similarity_with_problems(sol_id, sol_emb):
    mycursor = mydb.cursor()
    sql = "SELECT probId, probEmb FROM Problem"
    mycursor.execute(sql)
    rows = mycursor.fetchall()

    for row in rows:
        prob_id = row[0]
        prob_emb = row[1]
        score = ai.get_similarity(emb, prob_emb)
        insert_score(prob_id, sol_id, score)

def calc_similarity_with_solutions(prob_id, prob_emb):
    mycursor = mydb.cursor()
    sql = "SELECT solId, solEmb FROM Solution"
    mycursor.execute(sql)
    rows = mycursor.fetchall()

    for row in rows:
        sol_id = row[0]
        sol_emb = row[1]
        score = ai.get_similarity(emb, sol_emb)
        insert_score(prob_id, sol_id, score)

def insert_score(prob_id, sol_id, score):
    mycursor = mydb.cursor()
    sql = "INSERT INTO Score (probId, solId, score) VALUES (%s, %s, %s)"
    val = (prob_id, sol_id, score)
    mycursor.execute(sql, val)
    mydb.commit()



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

def get_embedding(raw_text, is_sol):
    response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user",
                           "content": prompts[is_sol] + f"{raw_text} Summarize the problems in no more than 150 words."}]
            )
    return encode_text_to_vector(response.choices[0].message.content)