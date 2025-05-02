import torch
import torch.nn as nn
from pathlib import Path
import json
from tqdm import tqdm

class NeuralEngine:
    def __init__(self, model_path="data/model_weights/model_weights.pth"):
        self.device = self._setup_device()
        self.model, self.vocab = self._load_model(model_path)
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}
        self._print_gpu_stats()

    def _setup_device(self):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        vocab = checkpoint["vocab"]
        
        config = checkpoint.get("config", {
            "model_params": {
                "embed_dim": 256,
                "hidden_dim": 512,
                "num_layers": 3,
                "dropout": 0.2
            }
        })
        
        model = TextGenerationModel(
            vocab_size=len(vocab),
            **config["model_params"]
        ).to(self.device)
        
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model, vocab

    def _print_gpu_stats(self):
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            print(f"\n💻 GPU: {props.name}")
            print(f"🔋 Всего памяти: {props.total_memory/1024**3:.1f} GB")
            print(f"⚡ Вычислительные ядра: {props.multi_processor_count}")
            print(f"🚀 Поддержка Tensor Cores: {'Да' if props.major >= 7 else 'Нет'}")

    def generate_response(self, text, max_length=50, temperature=0.7):
        tokens = [self.word2idx.get(word, 0) for word in text.split() if word in self.word2idx]
        if not tokens:
            return "Не могу обработать запрос. Попробуйте другие слова."

        input_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        generated = []
        with torch.no_grad():
            for _ in tqdm(range(max_length), desc="Генерация ответа"):
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs[0, -1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                if next_token == 0:
                    continue
                    
                generated.append(next_token)
                input_tensor = torch.cat([
                    input_tensor, 
                    torch.tensor([[next_token]], device=self.device)
                ], dim=1)

        return " ".join(self.idx2word[idx] for idx in generated if idx in self.idx2word)

class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x):
        embedded = self.embedding(x)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        output, _ = self.lstm(embedded, (h0, c0))
        return self.fc(output)