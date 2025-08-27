from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import openai
import os

app = Flask(__name__)
app.secret_key = "your_secret_key_here"   # Change to a secure random string

# Store users (for demo purpose only – use DB in production!)
users = {"test": "1234"}  # username: password

# Load OpenAI API key from env
openai.api_key = os.getenv("OPENAI_API_KEY")


# ---------------- AUTH ROUTES ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in users and users[username] == password:
            session["user"] = username
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in users:
            return render_template("signup.html", error="User already exists")
        users[username] = password
        session["user"] = username
        return redirect(url_for("home"))
    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ---------------- CHAT ROUTES ----------------
@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", user=session["user"])


@app.route("/chat", methods=["POST"])
def chat():
    if "user" not in session:
        return jsonify({"response": "⚠️ Please login first"})

    user_message = request.json.get("message")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI chatbot."},
                {"role": "user", "content": user_message}
            ]
        )
        bot_reply = response["choices"][0]["message"]["content"]
        return jsonify({"response": bot_reply})
    except Exception as e:
        return jsonify({"response": f"⚠️ Error: {str(e)}"})


if __name__ == "__main__":
    app.run(debug=True)