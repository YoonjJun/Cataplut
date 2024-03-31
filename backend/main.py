import csv
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
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request":request})

@app.get("/test")
async def test(request: Request):
    put_sample_data()
    return {}

@app.post("/company", response_class=HTMLResponse)
async def create_company(company_name: str = Form(...), problem: str = Form(...), solution: str = Form(...)):
    print("received")
    insert_company(company_name, problem, solution)
    matched = get_matched_companies(get_company_id(company_name))
    return templates.TemplateResponse("result.html", {"request":{}, "data": matched})

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
threshold = 0.85

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

    # print("company_id: ", company_id)

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

    # print("problem_id: ", prob_id)

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
        
        score = get_similarity(sol_emb, prob_emb)[0][0].item()
        # print("calc_score: ", score)
        if score > threshold:
            insert_score(prob_id, sol_id, score)

def calc_similarity_with_solutions(prob_id, prob_emb):
    mycursor = mydb.cursor()
    sql = "SELECT solId, solEmb FROM Solution"
    mycursor.execute(sql)
    rows = mycursor.fetchall()

    for row in rows:
        sol_id = row[0]
        sol_emb = np.frombuffer(row[1], dtype=np.float32).reshape(tensor_shape)
        sol_emb = torch.tensor(sol_emb, dtype=torch.float32)
        if prob_id == sol_id:
            continue
        score = get_similarity(prob_emb, sol_emb)[0][0].item()
        # print("calc_score: ", score)
        if score > threshold:
            insert_score(prob_id, sol_id, score)

def insert_score(prob_id, sol_id, score):
    mycursor = mydb.cursor()
    sql = "INSERT INTO Score (probId, solId, simScore) VALUES (%s, %s, %s)"
    val = (prob_id, sol_id, score)
    mycursor.execute(sql, val)
    mydb.commit()

    print("score inserted")

def get_company_name(company_id):
    mycursor = mydb.cursor()
    sql = "SELECT companyName FROM Company WHERE companyId = %s"
    val = (company_id,)
    row_count = mycursor.execute(sql, val)
    if row_count == 0:
        return None
    myresult = mycursor.fetchall()

    return myresult[0][0]

def get_company_id(name):
    mycursor = mydb.cursor()
    sql = "SELECT companyId FROM Company WHERE companyName = %s"
    val = (name,)
    mycursor.execute(sql, val)
    myresult = mycursor.fetchall()

    return myresult[0][0]

def get_company_id_of_sol(sol_id):
    mycursor = mydb.cursor()
    sql = "SELECT companyId FROM Solution WHERE solId = %s"
    val = (sol_id,)
    mycursor.execute(sql, val)
    myresult = mycursor.fetchall()

    return myresult[0][0]

def get_company_id_of_prob(prob_id):
    mycursor = mydb.cursor()
    sql = "SELECT companyId FROM Problem WHERE probId = %s"
    val = (prob_id,)
    mycursor.execute(sql, val)
    myresult = mycursor.fetchall()

    return myresult[0][0]

def get_matched_companies(company_id):
    prob_ids = get_problem_ids(company_id)
    # print("prob_id: ", prob_ids[0])
    sols_ids = get_solution_ids(company_id)
    # print("sol_id: ", sols_ids[0])
    mycursor = mydb.cursor()
    sql = "SELECT solId FROM Score WHERE probId = %s AND simScore > %s"
    val = (prob_ids[0], threshold)
    row_count = mycursor.execute(sql, val)
    if row_count == 0:
        return []
    myresult = mycursor.fetchall()
    matched_sols = [row[0] for row in myresult]

    matched_companies = []
    for sol_id in matched_sols:
        sol_company_id = get_company_id_of_sol(sol_id)
        prob_id_of_sol_company = get_problem_ids(sol_company_id)[0]
        score = get_score(prob_id_of_sol_company, sols_ids[0])
        # print("score: ", score)
        if score > threshold:
            print("matched")
            matched_companies.append({"company_name": get_company_name(sol_company_id)})
    
    if len(matched_companies) > 3:
        return matched_companies[:3]

    return matched_companies


def get_problem_ids(company_id):
    mycursor = mydb.cursor()
    sql = "SELECT probId FROM Problem WHERE companyId = %s"
    val = (company_id,)
    row_count = mycursor.execute(sql, val)
    if row_count == 0:
        return []
    myresult = mycursor.fetchall()
    prob_ids = [row[0] for row in myresult]

    return prob_ids

def get_solution_ids(company_id):
    mycursor = mydb.cursor()
    sql = "SELECT solId FROM Solution WHERE companyId = %s"
    val = (company_id,)
    row_count = mycursor.execute(sql, val)
    if row_count == 0:
        return []
    myresult = mycursor.fetchall()
    sol_ids = [row[0] for row in myresult]

    return sol_ids

def get_score(prob_id, sol_id):
    mycursor = mydb.cursor()
    sql = "SELECT simScore FROM Score WHERE probId = %s AND solId = %s"
    val = (prob_id, sol_id)
    row_count = mycursor.execute(sql, val)
    if row_count == 0:
        return 0.0
    myresult = mycursor.fetchall()
    if len(myresult) == 0:
        return 0.0

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
    # return encode_text_to_vector(raw_text)
    response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user",
                           "content": prompts[is_sol] + f"{raw_text} Summarize the problems in no more than 150 words."}]
            )
    return encode_text_to_vector(response.choices[0].message.content)

def get_similarity(emb1, emb2):
    return cosine_similarity(emb1, emb2)

def put_sample_data():
    with open('fake_company_data.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        i = 0
        for row in reader:
            if i < 1200:
                i += 1
                continue
            print(i)
            name = row[0]
            prob_text = row[1]
            sol_text = row[2]
            insert_company(name, prob_text, sol_text)
            i += 1
