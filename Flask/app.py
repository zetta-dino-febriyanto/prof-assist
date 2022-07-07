import os
from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
from haystack.nodes import PDFToTextConverter, PreProcessor
from haystack.preprocessor import PreProcessor
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.retriever import DensePassageRetriever
from haystack.reader import FARMReader
from haystack.pipeline import ExtractiveQAPipeline

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
        prof_assist_studio.document_file = os.path.join(app.config["UPLOAD_FOLDERS"], filename)
        pdf_converter = PDFToTextConverter(
            remove_numeric_tables=True, valid_languages=["en"])
        converted = pdf_converter.convert(
            file_path=prof_assist_studio.document_file, meta={"company": "Company_1", "processed": False})
        preprocessor = PreProcessor(split_by="word", split_length=200, split_overlap=10)
        preprocessed = preprocessor.process(converted)
        document_store = FAISSDocumentStore(
            faiss_index_factory_str="Flat", return_embedding=True)
        document_store.delete_all_documents()
        document_store.write_documents(preprocessed)
        retriever = DensePassageRetriever(document_store=document_store)
        reader = FARMReader(
            model_name_or_path="deepset/tinyroberta-squad2", use_gpu=False)
        document_store.update_embeddings(retriever)
        prof_assist_studio.pipeline = ExtractiveQAPipeline(reader, retriever)
        return render_template("prof-assist-studio.html")


def chatbot_response(msg):
    answers = []
    context = []
    for i in range(5):
        answers.append(get_prof_assist_response.prediction["answers"][i].answer)
        context.append(get_prof_assist_response.prediction["answers"][i].context)
    result = '<strong>Answer 1: </strong>' + '{}'.format(answers[0]) + '<br><br>' + '<strong>Context:</strong> ' + context[0] + '....' + '<br><br>' \
             '<strong>Answer 2: </strong>' +'{}'.format(answers[1]) + '<br><br>' + '<strong>Context:</strong> ' + context[1] + '....' + '<br><br>' \
             '<strong>Answer 3: </strong>' +'{}'.format(answers[2]) + '<br><br>' + '<strong>Context:</strong> ' + context[2] + '....' 
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
    os.remove('faiss_document_store.db')
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
