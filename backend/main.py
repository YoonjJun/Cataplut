from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
#템플릿 엔진 삽입
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="../templates")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/test")
async def test(request: Request):
    return templates.TemplateResponse("home.html", {"request":request})