from haystack.nodes import PDFToTextConverter, PreProcessor, QuestionGenerator
from haystack.preprocessor import PreProcessor
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.retriever import DensePassageRetriever
from haystack.reader import FARMReader
from haystack.pipelines import ExtractiveQAPipeline, QuestionAnswerGenerationPipeline
import tqdm
import xlsxwriter
import time


def preprocessing(file_path):
    pdf_converter = PDFToTextConverter(
        remove_numeric_tables=True, valid_languages=["en"])
    converted = pdf_converter.convert(file_path=file_path, meta={
                                      "company": "Company_1", "processed": False})
    preprocessor = PreProcessor(
        split_by="word", split_length=200, split_overlap=10)
    preprocessed = preprocessor.process(converted)
    return preprocessed


def document_store(document_preprocessed):
    document_store = FAISSDocumentStore(
        faiss_index_factory_str="Flat", return_embedding=True,)
    document_store.delete_all_documents()
    document_store.write_documents(document_preprocessed)
    return document_store


def question_answer_pipeline(document_store):
    retriever = DensePassageRetriever(document_store=document_store)
    reader = FARMReader(
        model_name_or_path='deepset/tinyroberta-squad2', use_gpu=True)
    document_store.update_embeddings(retriever)
    pipeline = ExtractiveQAPipeline(reader, retriever)
    return pipeline


def question_generator_pipeline(document_store):
    reader = FARMReader(
        model_name_or_path='deepset/tinyroberta-squad2', use_gpu=True)
    question_generator = QuestionGenerator()
    qag_pipeline = QuestionAnswerGenerationPipeline(question_generator, reader)
    return qag_pipeline


def question_generator(document_store, qag_pipeline):
    row = 1
    column = 0
    timestr = time.strftime("%Y%m%d-%H%M%S")
    question_generator.filename_excel = timestr + '.xls'
    workbook = xlsxwriter.Workbook('static/'+question_generator.filename_excel)
    worksheet = workbook.add_worksheet('Sheet 1')
    worksheet.write(0, 0, 'Question')
    worksheet.write(0, 1, 'Answer')
    worksheet.write(0, 2, 'Context')
    for idx, document in enumerate(tqdm.tqdm(document_store)):
        res = qag_pipeline.run(documents=[document])
        for i in range(0, len(res['queries'])):
            query = res['queries'][i]
            worksheet.write(row, column, query)

            answer = res['answers'][i][0].answer
            worksheet.write(row, column + 1, answer)

            contexts = res['answers'][i][0].context
            worksheet.write(row, column + 2, contexts)
            row += 1
    workbook.close()
