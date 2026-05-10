from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
 return "Hello! How are you chica?"

if __name__ == "main":
    app.run(debug=True)



