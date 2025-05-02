import spacy
import json
from pathlib import Path
from functools import lru_cache
from tqdm import tqdm

class TextProcessor:
    def __init__(self, lang='ru'):
        self.lang = lang
        self.config = self.load_config()
        self.nlp = self.load_spacy_model()
        self.slang = self.load_slang()
        self.vocab = self.load_vocab()

    def load_config(self):
        config_path = Path("data") / "config.json"
        if not config_path.exists():
            config_path.parent.mkdir(exist_ok=True)
            default_config = {
                "supported_langs": {
                    "ru": {
                        "model": "ru_core_news_sm",
                        "slang_file": "slang_ru.json",
                        "vocab_file": "vocab_ru.txt",
                        "max_text_length": 500000
                    }
                }
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            return default_config
        
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_spacy_model(self):
        nlp = spacy.load(self.config["supported_langs"][self.lang]["model"])
        nlp.max_length = self.config["supported_langs"][self.lang].get("max_text_length", 1000000)
        return nlp

    def load_slang(self):
        slang_file = Path("data") / self.config["supported_langs"][self.lang]["slang_file"]
        if not slang_file.exists():
            slang_file.parent.mkdir(exist_ok=True)
            with open(slang_file, "w", encoding="utf-8") as f:
                json.dump({}, f)
            return {}
        
        try:
            with open(slang_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def load_vocab(self):
        vocab_file = Path("data") / self.config["supported_langs"][self.lang]["vocab_file"]
        if not vocab_file.exists():
            vocab_file.parent.mkdir(exist_ok=True)
            with open(vocab_file, "w", encoding="utf-8") as f:
                f.write("")
            return []
        
        with open(vocab_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def process_text(self, text):
        """Обработка текста с разбиением на части при необходимости"""
        max_length = self.config["supported_langs"][self.lang].get("chunk_size", 200000)
        if len(text) > max_length:
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            processed_chunks = []
            for chunk in chunks:
                doc = self.nlp(chunk)
                processed_chunks.append(self._process_doc(doc))
            return " ".join(processed_chunks)
        else:
            doc = self.nlp(text)
            return self._process_doc(doc)

    def _process_doc(self, doc):
        """Внутренний метод обработки spaCy doc"""
        processed = []
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            lemma = self.slang.get(token.text.lower(), token.lemma_)
            processed.append(lemma)
        return " ".join(processed)

    def build_vocabulary(self, texts, min_word_count=3):
        """Автоматическое построение словаря из текстов"""
        vocab_file = Path("data") / self.config["supported_langs"][self.lang]["vocab_file"]
        word_counts = {}
        
        print("\n🔍 Анализ текстов для построения словаря...")
        for text in tqdm(texts, desc="Обработка текстов"):
            chunks = [text[i:i+200000] for i in range(0, len(text), 200000)]
            for chunk in chunks:
                doc = self.nlp(chunk)
                for token in doc:
                    if token.is_alpha and not token.is_stop and len(token.text) >= min_word_count:
                        lemma = token.lemma_.lower()
                        word_counts[lemma] = word_counts.get(lemma, 0) + 1
        
        vocab_file.parent.mkdir(exist_ok=True)
        with open(vocab_file, "w", encoding="utf-8") as f:
            for word in sorted(word_counts, key=word_counts.get, reverse=True):
                f.write(f"{word}\n")
        
        self.vocab = list(word_counts.keys())
        print(f"📚 Словарь создан. Всего слов: {len(self.vocab)}")
        return self.vocab