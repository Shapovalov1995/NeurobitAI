import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from pathlib import Path

class BookDataset(Dataset):
    def __init__(self, texts, vocab, seq_length=64):
        self.texts = texts
        self.vocab = vocab
        self.seq_length = seq_length
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        self.data = self._prepare_data()

    def _prepare_data(self):
        all_indices = []
        for text in self.texts:
            tokens = text.split()
            indices = [self.word2idx.get(token, 0) for token in tokens]
            all_indices.extend(indices)
        
        sequences = []
        for i in range(0, len(all_indices) - self.seq_length, self.seq_length):
            seq = all_indices[i:i + self.seq_length + 1]
            sequences.append(seq)
        
        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)

class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=3, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        
        if hidden is None:
            h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
            c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
            hidden = (h0, c0)
        
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden

class NeurobitTrainer:
    def __init__(self, config_path="data/config.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = self._load_config(config_path)
        self.vocab = self._load_vocab()
        self.model = self._init_model().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                   lr=self.config["training_params"]["learning_rate"])
        self.scaler = torch.cuda.amp.GradScaler()
        self._print_init_info()

    def _print_init_info(self):
        print(f"\n🛠 Инициализация NeurobitTrainer:")
        print(f"• Устройство: {self.device}")
        print(f"• Размер словаря: {len(self.vocab)} слов")
        print(f"• Параметры модели:")
        print(f"  - Embedding dim: {self.config['model_params']['embed_dim']}")
        print(f"  - Hidden dim: {self.config['model_params']['hidden_dim']}")
        print(f"  - LSTM layers: {self.config['model_params']['num_layers']}")
        
        if self.device.type == 'cuda':
            print(f"\n🎮 GPU-ускорение активно:")
            print(f"• Название: {torch.cuda.get_device_name(0)}")
            print(f"• Память: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    def _load_config(self, config_path):
        with open(config_path) as f:
            config = json.load(f)
        
        defaults = {
            "model_params": {
                "embed_dim": 256,
                "hidden_dim": 512,
                "num_layers": 3,
                "dropout": 0.2
            },
            "training_params": {
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 10,
                "seq_length": 64,
                "gradient_accumulation": 2
            }
        }
        
        for section in defaults:
            if section not in config:
                config[section] = defaults[section]
            else:
                for key in defaults[section]:
                    if key not in config[section]:
                        config[section][key] = defaults[section][key]
        
        return config

    def _load_vocab(self):
        vocab_file = Path("data") / self.config["supported_langs"]["ru"]["vocab_file"]
        if not vocab_file.exists():
            raise FileNotFoundError(f"Файл словаря не найден: {vocab_file}")
        
        with open(vocab_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _init_model(self):
        params = self.config["model_params"]
        model = TextGenerationModel(
            vocab_size=len(self.vocab),
            embed_dim=params["embed_dim"],
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            dropout=params["dropout"]
        )
        
        model = model.to(self.device)
        
        if self.device.type == 'cuda':
            model = torch.compile(model)
        
        return model

    def train_on_texts(self, texts, epochs=None):
        if not texts:
            print("⚠️ Нет текстов для обучения")
            return

        epochs = epochs or self.config["training_params"]["epochs"]
        dataset = BookDataset(texts, self.vocab, self.config["training_params"]["seq_length"])
        loader = DataLoader(
            dataset,
            batch_size=self.config["training_params"]["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )

        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0
            optimizer = self.optimizer
            
            with tqdm(loader, desc=f"Эпоха {epoch+1}/{epochs}") as pbar:
                for batch_idx, (inputs, targets) in enumerate(pbar):
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    with torch.cuda.amp.autocast():
                        outputs, _ = self.model(inputs)
                        loss = nn.CrossEntropyLoss()(
                            outputs.view(-1, len(self.vocab)),
                            targets.view(-1)
                        )
                    
                    self.scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % self.config["training_params"]["gradient_accumulation"] == 0:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    
                    total_loss += loss.item()
                    pbar.set_postfix(loss=total_loss/(batch_idx+1))

            avg_loss = total_loss / len(loader)
            print(f"Эпоха {epoch+1}, Средний loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model()

    def save_model(self, path="data/model_weights/model_weights.pth"):
        model_dir = Path(path).parent
        model_dir.mkdir(exist_ok=True, parents=True)
        
        torch.save({
            "model_state": self.model.state_dict(),
            "vocab": self.vocab,
            "config": self.config
        }, path)
        print(f"💾 Модель сохранена: {path}")

    def load_model(self, path="data/model_weights/model_weights.pth"):
        if not Path(path).exists():
            raise FileNotFoundError(f"Файл модели не найден: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.vocab = checkpoint["vocab"]
        self.config = checkpoint.get("config", self.config)
        print(f"♻️ Модель загружена из {path}")