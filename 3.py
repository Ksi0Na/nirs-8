import argparse
import queue
import sys
import sounddevice as sd
from vosk import Model, KaldiRecognizer


class VoiceRecognizer:
    def __init__(self,
                 model_path: str = "speech-to-text/vosk-model-small-ru-0.22"
                 ) -> None:
        self.model = Model(model_path)
        self.q = queue.Queue()

    @staticmethod
    def int_or_str(self, text):
        try:
            return int(text)
        except ValueError:
            return text

    def callback(self, indata, frames, time, status) -> None:
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))

    def recognize_speech(self, args) -> None:
        parser = self.create_parser()
        args = self.parse_arguments(parser, args)
        self.configure_samplerate(args)

        with sd.RawInputStream(samplerate=args.samplerate, blocksize=16000, device=args.device,
                               dtype="int16", channels=1, callback=self.callback):
            print("#" * 80)
            print("Press Ctrl+C to stop the recording")
            print("#" * 80)

            rec = KaldiRecognizer(self.model, args.samplerate)
            self.process_audio(rec, parser)

    def create_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("-l", "--list-devices", action="store_true", help="show list of audio devices and exit")
        return parser

    def parse_arguments(self, parser, args):
        args, remaining = parser.parse_known_args(args)
        if args.list_devices:
            print(sd.query_devices())
            parser.exit(0)

        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            parents=[parser]
        )
        parser.add_argument("-f", "--filename", type=str, metavar="FILENAME", help="audio file to store recording to")
        parser.add_argument("-d", "--device", type=self.int_or_str, help="input device (numeric ID or substring)")
        parser.add_argument("-r", "--samplerate", type=int, help="sampling rate")
        parser.add_argument("-m", "--model", type=str, help="language model; e.g. en-us, fr, nl; default is en-us")
        return parser.parse_args(remaining)

    def configure_samplerate(self, args) -> None:
        if args.samplerate is None:
            device_info = sd.query_devices(args.device, "input")
            args.samplerate = int(device_info["default_samplerate"])

    def process_audio(self, recognizer, parser) -> None:
        try:
            while True:
                data = self.q.get()
                if recognizer.AcceptWaveform(data):
                    print(recognizer.Result())

        except KeyboardInterrupt:
            print("\nDone")
            parser.exit(0)
        except Exception as e:
            parser.exit(type(e).__name__ + ": " + str(e))


if __name__ == '__main__':
    voice_recognizer = VoiceRecognizer()
    voice_recognizer.recognize_speech(sys.argv[1:])
