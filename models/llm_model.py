import os
from langchain.chains import ConversationalRetrievalChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate
from ipex_llm.langchain.llms import TransformersLLM
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from ipex_llm.langchain.embeddings import TransformersEmbeddings
from langchain import LLMChain
from utils.utils import new_cd

parent_dir = os.path.dirname(__file__)

condense_template = """
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the discussion is about the video content.
REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. \
Chat History:
{chat_history}
Follow Up Question: {question}
Standalone question:
"""

qa_template = """
You are an AI assistant designed for answering questions about a meeting.
You are given a word records of this meeting.
Try to comprehend the dialogs and provide a answer based on it.
=========
{context}
=========
Question: {question}
Answer: 
"""
# CONDENSE_QUESTION_PROMPT 用于将聊天历史记录和下一个问题压缩为一个独立的问题
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)
# QA_PROMPT为机器人设定基调和目的
QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["question", "context"])
# DOC_PROMPT = PromptTemplate.from_template("Video Clip {video_clip}: {page_content}")
DOC_PROMPT = PromptTemplate.from_template("{page_content}")


class LlmReasoner():
    def __init__(self, args):
        self.history = []
        self.llm_version = args.llm_version
        self.embed_version = args.embed_version
        self.qa_chain = None
        self.vectorstore = None
        self.top_k = args.top_k
        self.qa_max_new_tokens = args.qa_max_new_tokens
        self.init_model()

    def init_model(self):
        with new_cd(parent_dir):
            self.llm = TransformersLLM.from_model_id_low_bit(
                f"..\\checkpoints\\{self.llm_version}")
            self.llm.streaming = False
            self.embeddings = TransformersEmbeddings.from_model_id(
                model_id=f"..\\checkpoints\\{self.embed_version}")

    def create_qa_chain(self, args, input_log):
        self.top_k = args.top_k
        self.qa_max_new_tokens = args.qa_max_new_tokens
        self.question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT)
        self.answer_generator = LLMChain(llm=self.llm, prompt=QA_PROMPT,
                                         llm_kwargs={"max_new_tokens": self.qa_max_new_tokens})
        self.doc_chain = StuffDocumentsChain(llm_chain=self.answer_generator, document_prompt=DOC_PROMPT,
                                             document_variable_name='context')
        # 拆分查看字符的文本, 创建一个新的文本分割器
        # self.text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0, keep_separator=True)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=0)
        texts = self.text_splitter.split_text(input_log)
        self.vectorstore = FAISS.from_texts(texts, self.embeddings,
                                            metadatas=[{"video_clip": str(i)} for i in range(len(texts))])
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        self.qa_chain = ConversationalRetrievalChain(retriever=retriever,
                                                     question_generator=self.question_generator,
                                                     combine_docs_chain=self.doc_chain,
                                                     return_generated_question=True,
                                                     return_source_documents=True,
                                                     rephrase_question=False)

    def __call__(self, question):
        response = self.qa_chain({"question": question, "chat_history": self.history})
        answer = response["answer"]
        generated_question = response["generated_question"]
        source_documents = response["source_documents"]
        self.history.append([question, answer])
        return self.history, generated_question, source_documents

    def clean_history(self):
        self.history = []
