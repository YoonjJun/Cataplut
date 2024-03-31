from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import db

app = FastAPI()

templates = Jinja2Templates(directory="../templates")

@app.get("/")
def read_root():
    # db.test_db()
    return {"Hello": "World"}

@app.get("/test")
async def test(request: Request):
    return templates.TemplateResponse("home.html", {"request":request})