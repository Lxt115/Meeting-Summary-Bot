from bigdl.llm.langchain.llms import TransformersLLM
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain


def summarize(script):
    # file_path = "../test.txt"
    #
    # with open(file_path, "r", encoding="utf-8") as file:
    #     script = file.read()

    text_splitter = CharacterTextSplitter(chunk_size=10, separator="\n", chunk_overlap=0)
    texts = text_splitter.split_text(script)

    docs = [Document(page_content=t) for t in texts]

    llm = TransformersLLM.from_model_id_low_bit(f"D:\\Mcs\\5014\\llm\\models\\Llama-2-7b-chat-hf-INT4")
    # chain = load_summarize_chain(llm, chain_type="map_reduce", token_max=256)
    chain = load_summarize_chain(llm, chain_type="refine")
    # result = chain.run(docs)
    result = "pass"
    return result
