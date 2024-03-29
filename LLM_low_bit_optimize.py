from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer


llm = AutoModelForCausalLM.from_pretrained("D:\\Mcs\\5014\\llm\\models\\Llama-2-7b-chat-hf",
                                load_in_low_bit="sym_int4")
llm.save_low_bit("D:\\Mcs\\5014\\llm\\models\\Llama-2-7b-chat-hf-INT4")

tokenizer = LlamaTokenizer.from_pretrained("D:\\Mcs\\5014\\llm\\models\\Llama-2-7b-chat-hf\\")
tokenizer.save_pretrained("D:\\Mcs\\5014\\llm\\models\\Llama-2-7b-chat-hf-INT4")