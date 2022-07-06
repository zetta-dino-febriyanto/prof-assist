from flask import Flask, render_template, request

app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = set(['pdf'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prof-assist')
def prof_assist():
    return render_template('prof-assist.html')

def chatbot_response(msg):
    return "Hello, I'm ProfAssist. The Teacher's Digital Assistant for Students"

@app.route("/get")
def get_prof_assist_response():
    query = request.args.get('msg')
    return chatbot_response(query)

if __name__ == '__main__':
    app.run(debug=True)