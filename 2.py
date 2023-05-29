import os
import torch
import sounddevice as sd
import librosa
import wave
import time


class TextToSpeech:
    def __init__(self,
                 model_path: str = 'text-to-speech/ru/',
                 model_name: str = 'model.pt',
                 your_text: str = 'Приходится признать, что ни одна из этих сводок никуда не годится.'
                 ) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.your_text = your_text + ' ъ'
        self.device = torch.device('cpu')
        self.sample_rate = 48000
        self.speaker = 'eugene'
        self.model = None

    def download_model(self) -> None:
        if not os.path.isfile(self.model_path + self.model_name):
            os.makedirs(self.model_path, exist_ok=True)
            torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
                                           self.model_path + self.model_name)

    def load_model(self) -> None:
        self.model = torch.package.PackageImporter(self.model_path +
                                                   self.model_name).load_pickle("tts_models", "model")
        self.model.to(self.device)

    def save_text_to_speech(self) -> None:
        self.model.save_wav(text=self.your_text, speaker=self.speaker, sample_rate=self.sample_rate)

    @staticmethod
    def get_wav_duration(self, file_path) -> int:
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            print(duration)
            return int(duration)

    def play_audio_file(self, filename) -> None:
        data, sample_rate = librosa.load(filename, sr=None)
        sd.play(data, sample_rate)
        duration = self.get_wav_duration(filename)
        time.sleep(duration + 5)


if __name__ == '__main__':
    text_to_speech = TextToSpeech()
    text_to_speech.download_model()
    text_to_speech.load_model()

    filename = 'test.wav'
    text_to_speech.save_text_to_speech()
    text_to_speech.play_audio_file(filename)
