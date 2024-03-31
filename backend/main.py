from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import mysql.connector
from dotenv import load_dotenv
import os

from transformers import BertTokenizer, BertModel
import torch
from g4f.client import Client
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import nest_asyncio
nest_asyncio.apply()
app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")

templates = Jinja2Templates(directory="../templates")

class CompanyInfoRequest(BaseModel):
    company_name: str
    problem: str
    solution: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/test")
async def test(request: Request):
    test_db()
    return templates.TemplateResponse("index.html", {"request":request})

@app.post("/company")
async def create_company(company_name: str = Form(...), problem: str = Form(...), solution: str = Form(...)):
    print("received")
    insert_company(company_name, problem, solution)
    return {}

# @app.get("/matched/{company_name}")
# async def get_matched(company_name: str):
#     company_id = get_company_id(company_name)
#     # get matched company list
#     return {"matched_list": [{"company_name": company_name}]}

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

tensor_shape = (1, 768)

def test_db():
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM Company")
    myresult = mycursor.fetchall()
    for x in myresult:
        print(x)

def insert_company(name, prob_text, sol_text):
    mycursor = mydb.cursor()
    sql = "INSERT INTO Company (companyName) VALUES (%s)"
    val = (name,)
    mycursor.execute(sql, val)
    mydb.commit()

    print("company inserted")

    mycursor.execute("SELECT LAST_INSERT_ID()")
    company_id = mycursor.fetchone()[0]

    print("company_id: ", company_id)

    prob_emb = get_embedding(prob_text, 0)
    sol_emb = get_embedding(sol_text, 1)

    prob_id = insert_problem(company_id, prob_text, prob_emb)
    sol_id = insert_solution(company_id, sol_text, sol_emb)
    calc_similarity_with_solutions(prob_id, prob_emb)
    calc_similarity_with_problems(sol_id, sol_emb)
    

def insert_problem(company_id, prob_text, prob_emb):
    emb_bytes = prob_emb.numpy().tobytes()
    mycursor = mydb.cursor()
    sql = "INSERT INTO Problem (companyId, probText, probEmb) VALUES (%s, %s, %s)"
    val = (company_id, prob_text, emb_bytes)
    mycursor.execute(sql, val)
    mydb.commit()

    print("problem inserted")

    mycursor.execute("SELECT LAST_INSERT_ID()")
    prob_id = mycursor.fetchone()[0]

    print("problem_id: ", prob_id)

    return prob_id


def insert_solution(company_id, sol_text, sol_emb):
    emb_bytes = sol_emb.numpy().tobytes()
    mycursor = mydb.cursor()
    sql = "INSERT INTO Solution (companyId, solText, solEmb) VALUES (%s, %s, %s)"
    val = (company_id, sol_text, emb_bytes)
    mycursor.execute(sql, val)
    mydb.commit()

    print("solution inserted")

    mycursor.execute("SELECT LAST_INSERT_ID()")
    sol_id = mycursor.fetchone()[0]

    print("sol_id: ", sol_id)

    return sol_id

def calc_similarity_with_problems(sol_id, sol_emb):
    mycursor = mydb.cursor()
    sql = "SELECT probId, probEmb FROM Problem"
    mycursor.execute(sql)
    rows = mycursor.fetchall()

    for row in rows:
        prob_id = row[0]
        prob_emb = np.frombuffer(row[1], dtype=np.float32).reshape(tensor_shape)
        prob_emb = torch.tensor(prob_emb, dtype=torch.float32)
        
        score = get_similarity(sol_emb, prob_emb)
        insert_score(prob_id, sol_id, score[0][0].item())

def calc_similarity_with_solutions(prob_id, prob_emb):
    mycursor = mydb.cursor()
    sql = "SELECT solId, solEmb FROM Solution"
    mycursor.execute(sql)
    rows = mycursor.fetchall()

    for row in rows:
        sol_id = row[0]
        sol_emb = np.frombuffer(row[1], dtype=np.float32).reshape(tensor_shape)
        sol_emb = torch.tensor(sol_emb, dtype=torch.float32)
        score = get_similarity(prob_emb, sol_emb)
        insert_score(prob_id, sol_id, score[0][0].item())

def insert_score(prob_id, sol_id, score):
    mycursor = mydb.cursor()
    sql = "INSERT INTO Score (probId, solId, simScore) VALUES (%s, %s, %s)"
    val = (prob_id, sol_id, score)
    mycursor.execute(sql, val)
    mydb.commit()

def get_company_id(name):
    mycursor = mydb.cursor()
    sql = "SELECT companyId FROM Company WHERE companyName = %s"
    val = (name,)
    mycursor.execute(sql, val)
    myresult = mycursor.fetchall()

    return myresult[0][0]



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

def get_similarity(emb1, emb2):
    return cosine_similarity(emb1, emb2)