# -*- coding: utf-8 -*-
import argparse
import gradio as gr
import os
from models.helperbot_bigdl import Chat
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
parser.add_argument("--port", type=int, default=7860, help="Gradio server port")

args = parser.parse_args()

chat = Chat(args)
sumbot = Sum(args)
chat.init_model()

global_chat_history = []
global_result = ""

global_summary = ""


def clean_conversation():
    global global_chat_history
    chat.clean_history()
    global_chat_history = []
    return '', gr.update(value=None, interactive=True), None, gr.update(value=None, visible=True), gr.update(value=None,
                                                                                                             visible=True)


def clean_chat_history():
    global global_chat_history
    chat.clean_history()
    global_chat_history = []
    return '', None


def submit_message(message, max_tokens, top_p):
    args.qa_max_new_tokens = max_tokens
    args.top_k = top_p

    print(args)
    chat_history, generated_question, source_documents = chat.chat2video(args, message, global_result)
    global_chat_history.append((message, chat_history[0][1]))
    return '', global_chat_history


def gen_script(vid_path):
    print(vid_path)
    global global_result
    if vid_path is None:
        log_text = "===== Please upload video! ====="
        gr.update(value=log_text, visible=True)
    else:
        global_result = chat.video2log(vid_path)
        # script_pth = download_script_file()
        return gr.update(value=global_result, visible=True), download_script_file()


def download_script_file():
    try:
        with open("script_result.txt", "w") as file:
            file.write(global_result)
        return "script_result.txt"
    except Exception as e:
        return f"Error preparing file for download: {str(e)}"


def download_sum_file():
    try:
        with open("sum_result.txt", "w") as file:
            file.write(global_summary)
        return "sum_result.txt"
    except Exception as e:
        return f"Error preparing file for download: {str(e)}"


def upload_file(files):
    global global_result
    file_paths = [file.name for file in files][0]
    try:
        with open(file_paths, "r", encoding="utf-8") as file:
            file_content = file.read()
            global_result = file_content
    except FileNotFoundError:
        print("File not found")
    except IOError:
        print("Error occurred while reading the file")
    return file_content, download_script_file()


def summary():
    global global_summary
    global_summary = sumbot.summarize(global_result)
    return gr.update(value=global_summary, visible=True), download_sum_file()


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
        gr.Markdown(""" ## Meeting Helper Bot
        Upload meeting recording in mp3/mp4/txt format and you can get the summary and chat based on content  
        (You can adjust parameters based on your needs)
        Powered by BigDL, Llama, Whisper, and LangChain""",
                    elem_id="header")

        with gr.Column() as advanced_column:
            max_new_tokens = gr.Slider(label="Max new tokens", minimum=1, maximum=1024, step=1, value=128)
            top_k = gr.Slider(label="Top-k", minimum=1, maximum=50, step=1, value=3)

        with gr.Row():
            with gr.Column():
                video_inp = gr.Video(label="1.Upload MP3/MP4 File")
                # file_inp = gr.File(label="file/doc_input")
                upload_button = gr.UploadButton("1. Or Click to Upload a txt File", file_types=["doc", "txt"],
                                                file_count="multiple")
                gen_btn = gr.Button("2. Generate Script")
                sum_outp = gr.Textbox(label="Summerization output", lines=15)
                # save_sum_btn = gr.Button("Save Summarization to txt file")
                save_sum_dl = gr.outputs.File(label="Download Summary")
                # save_sum_btn.click(download_sum_file, [], outputs=[gr.outputs.File(label="Download Summary")])

            with gr.Column():
                script_outp = gr.Textbox(label="Script output", lines=30)
                with gr.Row():
                    script_summarization_btn = gr.Button("3.Script Summarization ")
                    # save_script_btn = gr.Button("Save Script to txt file")

                save_script_dl = gr.outputs.File(label="Download Script")
                # save_script_btn.click(download_script_file, [], outputs=[gr.outputs.File(label="Download Script")])

        with gr.Column():
            chatbot = gr.Chatbot(elem_id="chatbox")
            input_message = gr.Textbox(show_label=False, placeholder="Enter text and press enter", visible=True)
            btn_submit = gr.Button("Submit")
            with gr.Row():
                btn_clean_chat_history = gr.Button("Clean Chat History")
                btn_clean_conversation = gr.Button("Start New Conversation")

    upload_button.upload(upload_file, upload_button, [script_outp, save_script_dl])

    gen_btn.click(gen_script, [video_inp], [script_outp, save_script_dl])
    script_summarization_btn.click(summary, [], [sum_outp, save_sum_dl])

    btn_submit.click(submit_message, [input_message, max_new_tokens, top_k], [input_message, chatbot])
    input_message.submit(submit_message, [input_message, max_new_tokens, top_k], [input_message, chatbot])

    btn_clean_conversation.click(clean_conversation, [], [input_message, video_inp, chatbot, sum_outp, script_outp])
    btn_clean_chat_history.click(clean_chat_history, [], [input_message, chatbot])

    demo.load(queur=False)

demo.queue(concurrency_count=1)
demo.launch(height='800px', server_port=args.port, debug=True, share=False)
