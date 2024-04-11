from ipex_llm.langchain.llms import TransformersLLM
from langchain import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter


class Sum():
    def __init__(self, args):
        self.llm_version = args.llm_version
        # self.max_tokens = args.qa_max_new_tokens

    def summarize_refine(self, script):
        text_splitter = CharacterTextSplitter(chunk_size=1024, separator="\n", chunk_overlap=0)
        texts = text_splitter.split_text(script)
        docs = [Document(page_content=t) for t in texts]
        llm = TransformersLLM.from_model_id_low_bit(f"checkpoint\\{self.llm_version}")

        prompt_template = """Write a concise summary of the following:
        {text}
        CONCISE SUMMARY:"""
        prompt = PromptTemplate.from_template(prompt_template)
        refine_template = (
            "Your job is to produce a final summary\n"
            "We have provided an existing summary up to a certain point: {existing_answer}\n"
            "We have the opportunity to refine the existing summary"
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{text}\n"
            "------------\n"
            "If the context isn't useful, return the original summary."
        )
        refine_prompt = PromptTemplate.from_template(refine_template)
        chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            question_prompt=prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=True,
            input_key="input_documents",
            output_key="output_text",
        )
        result = chain({"input_documents": docs}, return_only_outputs=True)

        return result

    def summarize_mapreduce(self, script):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        texts = text_splitter.split_text(script)
        text = [Document(page_content=t) for t in texts]

        llm = TransformersLLM.from_model_id_low_bit(f"checkpoint\\{self.llm_version}")

        # Map
        map_template = """The following is a meeting recording
        =========
        {texts}
        =========
        Based on this list of recordings, please summary the main idea briefly
        Helpful Answer:"""
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=llm, prompt=map_prompt, llm_kwargs={"max_new_tokens": 512})

        # Reduce
        reduce_template = """The following is set of summaries:
        =========
        {texts}
        =========
        Take these and distill it into a final, consolidated summary of the meeting. 
        Helpful Answer:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt, llm_kwargs={"max_new_tokens": 4096})

        # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="texts"
        )

        # Combines and iteratively reduces the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=4000,
        )

        # Combining documents by mapping a chain over them, then combining results
        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="texts",
            return_intermediate_steps=False,
        )

        result = map_reduce_chain({"input_documents": text}, return_only_outputs=True)
        # print("-." * 40)
        # print(result)
        result = result['output_text'].split("Helpful Answer:").strip()[-1]
        return result

    def summarize(self, script):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
        texts = text_splitter.split_text(script)

        prompt_template = """The following is a piece of meeting recording:
        <<<{text}>>>
        Based on recording, summary the main idea fluently. 
        JUST SUMMARY!NO OTHER WORDS!
        SUMMARY:"""

        reduce_template = """The following is a meeting recording pieces:
        <<<{text}>>>
        Take these and distill it into a final, consolidated summary of the meeting. 
        JUST SUMMARY!NO OTHER WORDS!
        SUMMARY:"""

        print(len(texts))
        for text in texts:
            print(text)
            print("\n")

        llm = TransformersLLM.from_model_id_low_bit(
            f"checkpoint\\{self.llm_version}")
        sum_split = []

        for text in texts:
            response = llm(prompt=prompt_template.format(text=text), max_new_tokens=1024)
            print(response)
            response_answer = response.split("SUMMARY:")

            sum_split.append(response_answer[1])

        sum_all = "\n".join(sum_split)

        result = llm(prompt=reduce_template.format(text=sum_all), max_new_tokens=4000)
        result_split = result.split("SUMMARY:")
        return result_split[1]

# # for test
# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--llm_version", default="Llama-2-7b-chat-hf-INT4", help="LLM model version")
# args = parser.parse_args()
# file_path = "../test.txt"
# with open(file_path, "r", encoding="utf-8") as file:
#     content = file.read()
# Sumbot = Sum(args)
# result = Sumbot.summarize_map(content)
# print("-." * 20)
# print(result)
