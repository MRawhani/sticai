from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def index():
    print("Indexing")
    return {'data': {"name":'Mohammed', 'age':27}}