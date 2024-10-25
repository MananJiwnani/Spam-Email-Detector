from flask import Flask, render_template, request, redirect
import uuid
import model

app = Flask(__name__)

svc, tfidf_vectorizer = model.train_data()

@app.route('/', methods=["GET","POST"])
def home():
    if (request.method=="GET"):
        return render_template("home.html")
    elif (request.method=="POST"):
        email_content = request.form.get('email_text')
        result = model.training(email_content , svc, tfidf_vectorizer)
        return render_template("result.html", email_type = result, email=email_content)
    
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000)) 
    app.run(host='0.0.0.0', port=port)