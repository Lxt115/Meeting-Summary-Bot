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
parser.add_argument("--lid", default="en", choices=['en', 'zh'],
                    help="which language do you want use during conversation")

args = parser.parse_args()
print(args)

vchat = VChat(args)
vchat.init_model()

global_chat_history = []
global_lid = args.lid
global_en_log_result = ""
global_zh_log_result = ""


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


def submit_message(message, lid):
    chat_history, generated_question, source_documents = vchat.chat2video(message, lid)
    global_chat_history.append((message, chat_history[0][1]))
    return '', global_chat_history


def log_fn(vid_path, lid):
    print(vid_path)
    global global_en_log_result
    global global_zh_log_result
    if vid_path is None:
        log_text = "====== Please upload video or provide bilibili_BVid 🙃====="
        gr.update(value=log_text, visible=True)
    else:
        global_en_log_result, global_zh_log_result = vchat.video2log(vid_path)
        if lid == "en":
            return gr.update(value=global_en_log_result, visible=True)
        elif lid == "zh":
            return gr.update(value=global_zh_log_result, visible=True)


def lid_change(lid):
    global global_chat_history
    vchat.clean_history()
    global_chat_history = []
    print(f"\033[31;1mChange to {lid}\033[0m")
    if lid == "en":
        global global_en_log_result
        return gr.update(value=global_en_log_result, visible=True), '', None
    elif lid == "zh":
        global global_zh_log_result
        return gr.update(value=global_zh_log_result, visible=True), '', None


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
        lid_choice = gr.Dropdown(choices=["en", "zh"], value="en", label="Choose language")

        with gr.Row():
            with gr.Column():
                video_inp = gr.Video(label="video_input")
                log_btn = gr.Button("Generate Video Document")
                log_outp = gr.Textbox(label="Document output\nPlease be patient", lines=10)
                total_tokens_str = gr.Markdown(elem_id="total_tokens_str")

            with gr.Column():
                chatbot = gr.Chatbot(elem_id="chatbox")
                input_message = gr.Textbox(show_label=False, placeholder="Enter text and press enter",
                                           visible=True).style(container=False)
                btn_submit = gr.Button("Submit")
                with gr.Row():
                    btn_clean_chat_history = gr.Button("Clean Chat History")
                    btn_clean_conversation = gr.Button("🔃 Start New Conversation")
    # lid_choice 是一个下拉菜单控件，用于选择语言
    lid_choice.change(lid_change, [lid_choice], [log_outp, input_message, chatbot])
    # 用户可以在输入框 input_message 中输入文本f，并通过点击按钮 btn_submit 或按下回车键来提交消息
    btn_submit.click(submit_message, [input_message, lid_choice], [input_message, chatbot])
    input_message.submit(submit_message, [input_message, lid_choice], [input_message, chatbot])
    btn_clean_conversation.click(clean_conversation, [], [input_message, video_inp, chatbot, log_outp])
    btn_clean_chat_history.click(clean_chat_history, [], [input_message, chatbot])
    log_btn.click(log_fn, [video_inp, lid_choice], [log_outp])

    demo.load(queur=False)

demo.queue(concurrency_count=1)
demo.launch(height='800px', server_port=args.port, debug=True, share=True)
