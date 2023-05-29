from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline
import os


class TextGenerator:
    def __init__(self,
                 model_path: str = "text-gen/ru/small",
                 model_name: str = "ai-forever/rugpt3small_based_on_gpt2",
                 your_text: str = "Почему небо голубое"
                 ) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.your_text = your_text
        self.tokenizer = None
        self.model = None
        self.generator = None

    def download_model(self) -> None:
        if not os.path.exists(self.model_path):
            print(f"Downloading gpt-3 {self.model_name} model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelWithLMHead.from_pretrained(self.model_name)
            self.tokenizer.save_pretrained(self.model_path)
            self.model.save_pretrained(self.model_path)

    def load_generator(self) -> None:
        self.generator = pipeline(task='text-generation',
                                  model=self.model_path,
                                  tokenizer=self.model_path,
                                  framework='pt'
                                  )

    def generate_text(self) -> str:
        if self.generator is None:
            self.load_generator()
        gen_text = self.generator(text_inputs=self.your_text,
                                  do_sample=True,
                                  max_length=50,
                                  temperature=0.8)[0]['generated_text']
        return gen_text


if __name__ == '__main__':
    text_generator = TextGenerator()
    text_generator.download_model()
    text_generator.load_generator()
    generated_text = text_generator.generate_text()
    print(generated_text)
