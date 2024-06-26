import whisper
from ipex_llm import optimize_model

def has_intersection(t1, t2):
    if t1[1] < t2[0] or t2[1] < t1[0]:
        return False
    else:
        return True

class AudioTranslator():
    def __init__(self, args):
        self.model = whisper.load_model(args.whisper_version, download_root='checkpoints')
        self.model = optimize_model(self.model)

    def __call__(self, video_path):
        """
        input: video_path (str)
        output: audio_results (list)
        """
        print("Extract the audio results.")
        audio_results = self.model.transcribe(video_path, task = 'translate')["segments"]
        print("Finished.")
        return audio_results

    def match(self, audio_results):
        transcript = ''
        for res in audio_results:
            transcript += res['text'] + ' '
            # if has_intersection((start, end), (res["start"], res["end"])):
            #     transcript += res['text'] + ' '
        return transcript
