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
    '''
    This function is to Extract Text from PDF document
    and Preprocessing it

    :param str file_path: Path the PDF file
    '''
    #Convert PDF to Text
    pdf_converter = PDFToTextConverter(
        remove_numeric_tables=True, valid_languages=["en"])
    converted = pdf_converter.convert(file_path=file_path, meta={
                                      "company": "Company_1", "processed": False})

    #Preprocess the text and save it to preprocessor object
    preprocessor = PreProcessor(
        split_by="word", split_length=200, split_overlap=10)
    
    preprocessed = preprocessor.process(converted)

    #Return The 
    return preprocessed


def document_store(document_preprocessed):
    '''
    This function is to store document that has been preprocessed
    to the document store

    :param object document_preprocessed: Document that has been preprocessed
    '''
    #Initialization of Document Store
    document_store = FAISSDocumentStore(
        faiss_index_factory_str="Flat", return_embedding=True)

    #Delete all Documents on Document Store (to make sure is the document store is empty)
    document_store.delete_all_documents()

    #Store Preprocessed Text to out Document Store
    document_store.write_documents(document_preprocessed)

    #Return the Document Store
    return document_store


def question_answer_pipeline(document_store):
    '''
    This function is to make question answering pipeline

    :param object document_store: The document store
    '''
    #Initialize the Retriever
    retriever = DensePassageRetriever(document_store=document_store)

    #Initialize the Reader
    reader = FARMReader(
        model_name_or_path='deepset/tinyroberta-squad2', use_gpu=True)
    
    #Update embeddings on document store using Retriever
    document_store.update_embeddings(retriever)

    #Initialize the Pipeline using Reader and Retriever
    pipeline = ExtractiveQAPipeline(reader, retriever)

    #Return the pipeline
    return pipeline


def question_generator_pipeline(document_store):
    '''
    This function is to make question generation pipeline

    :param object document_store: The document store
    '''
    #Initialize the Reader
    reader = FARMReader(
        model_name_or_path='deepset/tinyroberta-squad2', use_gpu=True)

    #Initialize Question Generator Object
    question_generator = QuestionGenerator()

    #Initialize Question Generator Pipeline using question generator object and reader
    qag_pipeline = QuestionAnswerGenerationPipeline(question_generator, reader)

    #Return the question generator Pipeline
    return qag_pipeline


def question_generator(document_store, qag_pipeline):
    '''
    This function is to perform question generator (to generate question from document on document store)
    and save it to excel

    :param object document_store: The document store
    :param object qag_pipeline: Question generator pipeline
    '''
    #Initialize Variable
    row = 1
    column = 0
    timestr = time.strftime("%Y%m%d-%H%M%S")
    question_generator.filename_excel = timestr + '.xls'

    #Make excel file
    workbook = xlsxwriter.Workbook('static/'+question_generator.filename_excel)
    #Add sheet on excel file
    worksheet = workbook.add_worksheet('Sheet 1')
    #Add Data on excel file
    worksheet.write(0, 0, 'Question')
    worksheet.write(0, 1, 'Answer')
    worksheet.write(0, 2, 'Context')

    #Perform Question Generator
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
