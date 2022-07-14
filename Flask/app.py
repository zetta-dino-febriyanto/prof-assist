import os
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import module as md

# Initialize the Flask application
app = Flask(__name__)

# Define the allowed extensions and upload directory
app.config["ALLOWED_EXTENSIONS"] = set(["pdf"])
app.config["UPLOAD_FOLDERS"] = "uploads/"

# A function to check the file extension
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]
    )

# A index route
@app.route("/")
def index():
    return render_template("index.html")

# A route to question answering page
@app.route('/question-answering')
def question_answering():
    return render_template('question-answering.html')

# A route to question generation page
@app.route('/question-generator')
def question_generator():
    return render_template('question-generator.html')

# A route to process the PDF document and
# get the response from the chatbot
@app.route('/prof-assist-studio', methods=['GET', 'POST'])
def prof_assist_studio():
    document = request.files["file"] # Get the file from the request
    if document and allowed_file(document.filename): # Check if the file is allowed
        filename = secure_filename(document.filename) # Get the filename
        document.filename.replace(' ', '_') # Replace spaces with underscores
        document.save(os.path.join(app.config["UPLOAD_FOLDERS"], filename)) # Save the file
        prof_assist_studio.document_file = os.path.join(
            app.config["UPLOAD_FOLDERS"], filename) # Save the file path
        preprocessed = md.preprocessing(prof_assist_studio.document_file) # Preprocess the document
        document_store = md.document_store(preprocessed) # Store the preprocessed document
        prof_assist_studio.pipeline = md.question_answer_pipeline(document_store) # Running the question answering pipeline
    return render_template("prof-assist-studio.html") # A route to chatbot room page


# A route to get the question generator result
# and download it as an Excel file
@app.route('/questions-result', methods=['GET', 'POST'])
def questions_result():
    document = request.files["file"] # Get the document
    if document and allowed_file(document.filename): # Check if the document is valid
        filename = secure_filename(document.filename) # Get the filename
        document.filename.replace(' ', '_') # Replace the spaces with underscores
        document.save(os.path.join(app.config["UPLOAD_FOLDERS"], filename)) # Save the document
        questions_result.document_file = os.path.join(
            app.config["UPLOAD_FOLDERS"], filename) # Store the document path
        preprocessed = md.preprocessing(questions_result.document_file) # Preprocess the document
        document_stores = md.summarization(preprocessed) # Summarize the Document
        qag_pipeline = md.question_generator_pipeline(document_stores) # Get the question generator pipeline
        file_excel = md.question_generator(document_stores, qag_pipeline) # Get the questions generated
    return render_template("questions-result.html", file_excel=file_excel) # A route to get the question generator result


@app.route('/download_excel/<excel>')
def download_file(excel):
  excel = 'static/'+excel
  print(excel)
  return send_file(excel, as_attachment=True) # Return download excel file


def chatbot_response(msg):
    answers = [] # Initialize the answers list
    context = [] # Initialize the context list
    for i in range(5): # Get the 5 most recent answers
        answers.append(
            get_prof_assist_response.prediction["answers"][i].answer) # Get the answer and append it to the answers list
        context.append(
            get_prof_assist_response.prediction["answers"][i].context) # Get the context and append it to the context list
    # Get the result of the top 3 answers
    # and also return the context of each answer
    result = \
        '<strong>Answer 1: </strong>' + '{}'.format(answers[0]) + '<br><br>' + '<strong>Context:</strong> ' + context[0] + '....' + '<br><br>' \
        '<strong>Answer 2: </strong>' + '{}'.format(answers[1]) + '<br><br>' + '<strong>Context:</strong> ' + context[1] + '....' + '<br><br>' \
        '<strong>Answer 3: </strong>' + '{}'.format(answers[2]) + '<br><br>' + '<strong>Context:</strong> ' + context[2] + '....' 
    return result # Return the result


@app.route("/get")
def get_prof_assist_response():
    query = request.args.get("msg") # Get the query
    if query == 'Hello': # Check if the query is hello
        return "Hello! This is ProfAssist, the teacher's assistant for Students." # Return the response
    get_prof_assist_response.prediction = prof_assist_studio.pipeline.run(
        query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 10}}
    ) # Get the prediction
    return chatbot_response(query) # Return the response


@app.route("/done")
def done():
    return render_template("index.html") # Return to the index page


if __name__ == "__main__":
    app.run(debug=True)
