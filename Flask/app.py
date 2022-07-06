from flask import Flask, render_template, request
from haystack.nodes import PDFToTextConverter, PreProcessor
from haystack.preprocessor import PreProcessor
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.retriever import DensePassageRetriever
from haystack.reader import FARMReader
from haystack.pipeline import ExtractiveQAPipeline

app = Flask(__name__)

app.config["ALLOWED_EXTENSIONS"] = set(["pdf"])


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prof-assist")
def prof_assist():
    return render_template("prof-assist.html")


def document_extractor():
    document = request.files["file"]
    if document and allowed_file(document.filename):
        pdf_converter = PDFToTextConverter(
            remove_numeric_tables=True, valid_languages=["en"]
        )
        converted = pdf_converter.convert(
            file_path=document, meta={"company": "Company_1", "processed": False}
        )
        preprocessor = PreProcessor(split_by="word", split_length=200, split_overlap=10)
        preprocessed = preprocessor.process(converted)
        document_store = FAISSDocumentStore(
            faiss_index_factory_str="Flat", return_embedding=True
        )
        document_store.delete_all_documents()
        document_store.write_documents(preprocessed)
        retriever = DensePassageRetriever(document_store=document_store)
        reader = FARMReader(
            model_name_or_path="deepset/tinyroberta-squad2", use_gpu=True
        )
        document_store.update_embeddings(retriever)
        document_extractor.pipeline = ExtractiveQAPipeline(reader, retriever)
        return render_template("prof-assist.html")


def chatbot_response(msg):
    return get_prof_assist_response.prediction["answers"].answer
    # return "Hello, I'm ProfAssist. The Teacher's Digital Assistant for Students"


@app.route("/get")
def get_prof_assist_response():
    query = request.args.get("msg")
    get_prof_assist_response.prediction = document_extractor.pipeline.run(
        query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 10}}
    )
    return chatbot_response(query)


if __name__ == "__main__":
    app.run(debug=True)
