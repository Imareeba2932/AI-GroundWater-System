from flask import Flask, render_template, request, redirect, url_for, session, flash
import plotly.express as px
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "secret123"

users = {}

# DATA
if os.path.exists('groundwater_ml_dataset_cleaned.csv'):
    df = pd.read_csv('groundwater_ml_dataset_cleaned.csv')
else:
    df = pd.DataFrame({'category': ['Safe', 'Critical', 'Overexploited']})

# HOME
@app.route('/')
def home():
    return render_template('index.html')

# AUTH
@app.route('/auth', methods=['GET', 'POST'])
def auth():
    mode = request.args.get('mode', 'register')

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # ================= REGISTER =================
        if mode == 'register':

            if email in users:
                flash("⚠ You are already registered! Please login.", "error")
                return redirect(url_for('auth', mode='login'))

            users[email] = password

            flash("✅ Registered successfully! Please login.", "success")
            return redirect(url_for('auth', mode='login'))

        # ================= LOGIN =================
        if mode == 'login':

            if email not in users:
                flash("❌ User not found! Please register first.", "error")
                return redirect(url_for('auth', mode='register'))

            if users[email] == password:

                session['user_email'] = email
                session['user_name'] = email.split('@')[0]

                flash("🎉 Login successful!", "success")
                return redirect(url_for('home'))
            else:
                flash("❌ Wrong password!", "error")
                return redirect(url_for('auth', mode='login'))

    return render_template('auth.html', mode=mode)

# PROTECTION
def login_required():
    return 'user_email' not in session

@app.route('/about')
def about():
    if login_required():
        return redirect(url_for('auth', mode='login'))
    return render_template('about.html')

@app.route('/overview')
def overview():
    if login_required():
        return redirect(url_for('auth', mode='login'))

    fig = px.pie(
        df,
        names='category',
        title='Category Distribution',
        color_discrete_sequence=px.colors.sequential.Tealgrn
    )

    graph = fig.to_html(full_html=False)
    return render_template('overview.html', graph=graph)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if login_required():
        return redirect(url_for('auth', mode='login'))

    result = None
    if request.method == 'POST':
        rainfall = float(request.form.get('rainfall'))
        extraction = float(request.form.get('extraction'))
        result = round((rainfall * 0.6) - (extraction * 0.4), 2)

    return render_template('prediction.html', prediction=result)

@app.route('/contact')
def contact():
    return render_template('contact.html')

# ✅ DASHBOARD ROUTE
@app.route('/dashboard')
def dashboard():
    if login_required():
        return redirect(url_for('auth', mode='login'))
    return render_template('dashboard.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully!", "success")
    return redirect(url_for('home'))

# =========================
# 🤖 AI CHAT SUPPORT (WORKING)
# =========================
@app.route('/ai-chat', methods=['POST'])
def ai_chat():
    user_msg = request.form.get('message', '').lower()

    response = "Sorry, I didn't understand that. Try asking about prediction, dataset, or groundwater."

    if "prediction" in user_msg:
        response = "AI predicts groundwater using rainfall and extraction data with ML logic."

    elif "dataset" in user_msg:
        response = "We use historical groundwater CSV dataset for training and visualization."

    elif "help" in user_msg:
        response = "You can ask about prediction, dataset, overview, or system usage."

    elif "hello" in user_msg or "hi" in user_msg:
        response = "Hello 👋 How can I help you today?"

    elif "groundwater" in user_msg:
        response = "Groundwater is analyzed based on extraction vs rainfall trends."

    return {"reply": response}


if __name__ == '__main__':
    app.run(debug=True)
