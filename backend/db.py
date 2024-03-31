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