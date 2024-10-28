from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/detect/{title}")
def detect_news(title: str):
    return {"title": title}