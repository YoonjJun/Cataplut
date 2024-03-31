import mysql.connector
from dotenv import load_dotenv
import os

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