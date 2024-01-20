from bigdl.llm.langchain.llms import TransformersLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Sum():
    def __init__(self, args):
        self.llm_version = args.llm_version
        # self.max_tokens = args.qa_max_new_tokens

    def summarize(self, script):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1300, chunk_overlap=0)
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

        llm = TransformersLLM.from_model_id_low_bit(f"D:\\Mcs\\5014\\VChat-BigDL\\checkpoint\\{self.llm_version}")
        sum_split = []

        for text in texts:
            response = llm(prompt=prompt_template.format(text=text), max_new_tokens=1000)
            print(response)
            response_answer = response.split("SUMMARY:")

            sum_split.append(response_answer[1])

        sum_all = "\n".join(sum_split)

        result = llm(prompt=reduce_template.format(text=sum_all),  max_new_tokens=4000)
        result_split = result.split("SUMMARY:")
        return result_split[1]


# for test
# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--llm_version", default="Llama-2-7b-chat-hf-INT4", help="LLM model version")
# args = parser.parse_args()
# file_path = "../test.txt"
# with open(file_path, "r", encoding="utf-8") as file:
#     content = file.read()
# Sumbot = Sum(args)
# result = Sumbot.summarize(content)
#
# print(result)
