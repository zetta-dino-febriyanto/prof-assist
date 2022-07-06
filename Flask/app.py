from flask import Flask, render_template

app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = set(['pdf'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prof-assist')
def prof_assist():
    return render_template('prof-assist.html')

if __name__ == '__main__':
    app.run(debug=True)