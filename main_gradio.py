# -*- coding: utf-8 -*-
import argparse
import gradio as gr
import os
from models.vchat_bigdl import VChat
from models.sum_model import Sum

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
sumbot = Sum(args)
vchat.init_model()

global_chat_history = []
global_en_log_result = ""
global_summary = ""


def clean_conversation():
    global global_chat_history
    vchat.clean_history()
    global_chat_history = []
    return '', gr.update(value=None, interactive=True), None, gr.update(value=None, visible=True), gr.update(value=None, visible=True)


def clean_chat_history():
    global global_chat_history
    vchat.clean_history()
    global_chat_history = []
    return '', None


def submit_message(message):
    print(args)
    chat_history, generated_question, source_documents = vchat.chat2video(message)
    global_chat_history.append((message, chat_history[0][1]))
    return '', global_chat_history


def gen_script(vid_path):
    print(vid_path)
    global global_en_log_result
    if vid_path is None:
        log_text = "===== Please upload video! ====="
        gr.update(value=log_text, visible=True)
    else:
        global_en_log_result = vchat.video2log(vid_path)
        return gr.update(value=global_en_log_result, visible=True)


def download_script_file():
    try:
        with open("script_result.txt", "w") as file:
            file.write(global_en_log_result)
        # return "temp_en_log_result.txt"
    except Exception as e:
        return f"Error preparing file for download: {str(e)}"

def download_sum_file():
    try:
        with open("sum_result.txt", "w") as file:
            file.write(global_summary)
        # return "temp_en_log_result.txt"
    except Exception as e:
        return f"Error preparing file for download: {str(e)}"

def summary():
    global global_summary
    global_summary = sumbot.summarize(global_en_log_result)
    return gr.update(value=global_summary, visible=True)


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
            # temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.0, step=0.1, value=1.0)
            # top_p = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.95)
            top_k = gr.Slider(label="Top-k", minimum=1, maximum=50, step=1, value=3)


            args.qa_max_new_tokens = max_new_tokens
            args.top_k = top_k

        with gr.Row():
            with gr.Column():
                video_inp = gr.Video(label="video/mp3_input")
                gen_btn = gr.Button("Generate Script")
                sum_outp = gr.Textbox(label="Summerization output\nPlease be patient", lines=15)
                save_sum_btn = gr.Button("Save Summarization to txt file")

            with gr.Column():
                script_outp = gr.Textbox(label="Script output\nPlease be patient", lines=30)
                with gr.Row():
                    script_summarization_btn = gr.Button("Script Summarization ")
                    save_script_btn = gr.Button("Save Script to txt file")

        with gr.Column():
            chatbot = gr.Chatbot(elem_id="chatbox")
            input_message = gr.Textbox(show_label=False, placeholder="Enter text and press enter", visible=True)
            btn_submit = gr.Button("Submit")
            with gr.Row():
                btn_clean_chat_history = gr.Button("Clean Chat History")
                btn_clean_conversation = gr.Button("Start New Conversation")

    gen_btn.click(gen_script, [video_inp], [script_outp])
    script_summarization_btn.click(summary, [], [sum_outp])
    save_sum_btn.click(download_sum_file, [], outputs=[gr.outputs.File(label="Download Summary")])
    save_script_btn.click(download_script_file, [], outputs=[gr.outputs.File(label="Download Script")])

    btn_submit.click(submit_message, [input_message], [input_message, chatbot])
    input_message.submit(submit_message, [input_message], [input_message, chatbot])
    btn_clean_conversation.click(clean_conversation, [], [input_message, video_inp, chatbot, sum_outp, script_outp])
    btn_clean_chat_history.click(clean_chat_history, [], [input_message, chatbot])

    demo.load(queur=False)

demo.queue(concurrency_count=1)
demo.launch(height='800px', server_port=args.port, debug=True, share=True)
