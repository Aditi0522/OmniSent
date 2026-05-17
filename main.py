from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def hello():
 return "Hello! How are you chica?"

if __name__ == "__main__":
    app.run(debug=True)



