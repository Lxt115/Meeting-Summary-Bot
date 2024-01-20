from models.whisper_model import AudioTranslator
from models.llm_model import LlmReasoner


class VChat:

    def __init__(self, args) -> None:
        self.args = args

    def init_model(self):
        print('\033[1;33m' + "Initializing models...".center(50, '-') + '\033[0m')
        self.audio_translator = AudioTranslator(self.args)
        self.llm_reasoner = LlmReasoner(self.args)

        print('\033[1;32m' + "Model initialization finished!".center(50, '-') + '\033[0m')

    def video2log(self, video_path):
        audio_results = self.audio_translator(video_path)

        en_log_result = []
        en_log_result_tmp = ""
        audio_transcript = self.audio_translator.match(audio_results)
        en_log_result_tmp += f"\n{audio_transcript}"

        en_log_result.append(en_log_result_tmp)

        en_log_result = "\n\n".join(en_log_result)
        print(f"\033[1;34mLog: \033[0m\n{en_log_result}\n")

        return en_log_result

    def chat2video(self, args, user_input, en_log_result):
        self.llm_reasoner.create_qa_chain(args, en_log_result)
        en_user_input = user_input

        print("\n\033[1;32mGnerating response...\033[0m")
        answer, generated_question, source_documents = self.llm_reasoner(en_user_input)
        print(f"\033[1;32mQuestion: \033[0m{user_input}")
        print(f"\033[1;32mAnswer: \033[0m{answer[0][1]}")
        self.clean_history()

        return answer, generated_question, source_documents

    def clean_history(self):
        self.llm_reasoner.clean_history()
        return
