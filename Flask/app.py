import os
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import module as md

app = Flask(__name__)

app.config["ALLOWED_EXTENSIONS"] = set(["pdf"])
app.config["UPLOAD_FOLDERS"] = "uploads/"


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/question-answering')
def question_answering():
    return render_template('question-answering.html')


@app.route('/question-generator')
def question_generator():
    return render_template('question-generator.html')


@app.route('/prof-assist-studio', methods=['GET', 'POST'])
def prof_assist_studio():
    document = request.files["file"]
    if document and allowed_file(document.filename):
        filename = secure_filename(document.filename)
        document.filename.replace(' ', '_')
        document.save(os.path.join(app.config["UPLOAD_FOLDERS"], filename))
        prof_assist_studio.document_file = os.path.join(
            app.config["UPLOAD_FOLDERS"], filename)
        preprocessed = md.preprocessing(prof_assist_studio.document_file)
        document_store = md.document_store(preprocessed)
    return render_template("prof-assist-studio.html")


@app.route('/questions-result', methods=['GET', 'POST'])
def questions_result():
    document = request.files["file"]
    if document and allowed_file(document.filename):
        filename = secure_filename(document.filename)
        document.filename.replace(' ', '_')
        document.save(os.path.join(app.config["UPLOAD_FOLDERS"], filename))
        questions_result.document_file = os.path.join(
            app.config["UPLOAD_FOLDERS"], filename)
        preprocessed = md.preprocessing(questions_result.document_file)
        document_stores = md.document_store(preprocessed)
        qag_pipeline = md.question_generator_pipeline(document_stores)
        question_generator(document_stores, qag_pipeline)
    return render_template("questions-result.html")


@app.route('/download_excel')
def download_file():
    excel = 'static/'+question_generator.filename_excel
    return send_file(excel, as_attachment=True)


def chatbot_response(msg):
    answers = []
    context = []
    for i in range(5):
        answers.append(
            get_prof_assist_response.prediction["answers"][i].answer)
        context.append(
            get_prof_assist_response.prediction["answers"][i].context)
    result = \
        '<strong>Answer 1: </strong>' + '{}'.format(answers[0]) + '<br><br>' + '<strong>Context:</strong> ' + context[0] + '....' + '<br><br>' \
        '<strong>Answer 2: </strong>' + '{}'.format(answers[1]) + '<br><br>' + '<strong>Context:</strong> ' + context[1] + '....' + '<br><br>' \
        '<strong>Answer 3: </strong>' + '{}'.format(answers[2]) + '<br><br>' + '<strong>Context:</strong> ' + context[2] + '....' 
    return result


@app.route("/get")
def get_prof_assist_response():
    query = request.args.get("msg")
    if query == 'Hello':
        return "Hello! This is ProfAssist, the teacher's assistant for Students."
    get_prof_assist_response.prediction = prof_assist_studio.pipeline.run(
        query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 10}}
    )
    return chatbot_response(query)


@app.route("/done")
def done():
    try:
        os.remove('faiss_document_store.db')
    except:
        pass
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
