# -*- coding: utf-8 -*-
import argparse
import gradio as gr
import os
from models.vchat_bigdl import VChat

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
parser = argparse.ArgumentParser()

# whisper model arguments
parser.add_argument("--whisper_version", default="small", help="Whisper model version for video asr")
# llm model arguments
parser.add_argument("--llm_version", default="Llama-2-7b-chat-hf-INT4", help="LLM model version")
parser.add_argument("--embed_version", default="all-MiniLM-L12-v2", help="Embedding model version")
parser.add_argument("--top_k", default=3, type=int, help="Return top k relevant contexts to llm")
parser.add_argument("--qa_max_new_tokens", default=128, type=int, help="Number of max new tokens for llm")
# general arguments
parser.add_argument("--port", type=int, default=8899, help="Gradio server port")

args = parser.parse_args()
print(args)

vchat = VChat(args)
vchat.init_model()

global_chat_history = []
global_en_log_result = ""


def clean_conversation():
    global global_chat_history
    vchat.clean_history()
    global_chat_history = []
    return '', gr.update(value=None, interactive=True), None, gr.update(value=None, visible=True)


def clean_chat_history():
    global global_chat_history
    vchat.clean_history()
    global_chat_history = []
    return '', None


def submit_message(message):
    chat_history, generated_question, source_documents = vchat.chat2video(message)
    global_chat_history.append((message, chat_history[0][1]))
    return '', global_chat_history


def log_fn(vid_path):
    print(vid_path)
    global global_en_log_result
    if vid_path is None:
        log_text = "===== Please upload video! ====="
        gr.update(value=log_text, visible=True)
    else:
        global_en_log_result = vchat.video2log(vid_path)
        return gr.update(value=global_en_log_result, visible=True)
        

def download_file():
    with open("en_log_result.txt", "w") as file:
        file.write(global_en_log_result)

css = """
      #col-container {max-width: 80%; margin-left: auto; margin-right: auto;}
      #video_inp {min-height: 100px}
      #chatbox {min-height: 100px;}
      #header {text-align: center;}
      #hint {font-size: 1.0em; padding: 0.5em; margin: 0;}
      .message { font-size: 1.2em; }
      """

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("""## 🤖SumMeeting Bot
                    Powered by BigDL, Llama, Whisper, Helsinki and LangChain/log""",
                    elem_id="header")
        

        with gr.Column() as advanced_column:
            max_new_tokens = gr.Slider(label="Max new tokens", minimum=1, maximum=1024, step=1, value=128)
            temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.0, step=0.1, value=1.0)
            top_p = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.95)
            top_k = gr.Slider(label="Top-k", minimum=1, maximum=50, step=1, value=3)

        with gr.Row():
            with gr.Column():
                video_inp = gr.Video(label="video_input")
                log_btn = gr.Button("Generate Video Document")
                log_outp = gr.Textbox(label="Summary Preview", lines=15)
                btn_download = gr.Button("Download File")
                # total_tokens_str = gr.Markdown(elem_id="total_tokens_str")

            with gr.Column():
                chatbot = gr.Chatbot(elem_id="chatbox", height=600)
                input_message = gr.Textbox(show_label=False, placeholder="Enter text and press enter",
                                           visible=True).style(container=False)
                btn_submit = gr.Button("Submit")
                with gr.Row():
                    btn_clean_chat_history = gr.Button("Clean Chat History")
                    btn_clean_conversation = gr.Button("Start New Summary")

    btn_submit.click(submit_message, [input_message, max_new_tokens, temperature, top_p, top_k],
                     [input_message, chatbot])
    input_message.submit(submit_message, [input_message], [input_message, chatbot])
    btn_clean_conversation.click(clean_conversation, [], [input_message, video_inp, chatbot, log_outp])
    btn_clean_chat_history.click(clean_chat_history, [], [input_message, chatbot])
    log_btn.click(log_fn, [video_inp], [log_outp])
    btn_download.click(download_file, [])

    demo.load(queur=False)

demo.queue(concurrency_count=1)
demo.launch(height='800px', server_port=args.port, debug=True, share=True)
