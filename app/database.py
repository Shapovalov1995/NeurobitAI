import hashlib
import json
from pathlib import Path
from datetime import datetime
from app.book_parser import parse_book

class Database:
    def __init__(self, db_path="data/books"):
        self.db_path = Path(db_path)
        self.archive_path = self.db_path.parent / "archive"
        self.archive_path.mkdir(exist_ok=True)
        self.processing_log = self.db_path.parent / "processing.json"
        
        if not self.db_path.exists():
            self.db_path.mkdir(parents=True)

    def clean_log(self):
        """Очищает лог от записей о несуществующих файлах"""
        if not self.processing_log.exists():
            return 0
            
        try:
            with open(self.processing_log) as f:
                log_data = json.load(f)
        except json.JSONDecodeError:
            return 0
            
        original_count = len(log_data)
        updated_log = {
            str(path): hash_val for path, hash_val in log_data.items() 
            if Path(path).exists()
        }
        
        with open(self.processing_log, "w") as f:
            json.dump(updated_log, f, indent=2)
            
        return original_count - len(updated_log)

    def clean_orphaned_entries(self):
        """Алиас для clean_log()"""
        return self.clean_log()

    def _get_file_hash(self, filepath):
        """Вычисляет MD5-хеш файла"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _get_content_hash(self, filepath):
        """Хеш содержимого книги"""
        try:
            text = parse_book(filepath)
            return hashlib.md5(text.encode('utf-8')).hexdigest()
        except Exception as e:
            print(f"⚠️ Ошибка хеширования {filepath.name}: {str(e)}")
            return None

    def get_all_books(self):
        """Рекурсивный поиск книг во всех подпапках"""
        pdfs = list(self.db_path.rglob("*.pdf"))
        epubs = list(self.db_path.rglob("*.epub"))
        return pdfs + epubs

    def load_processing_log(self):
        """Загружает лог обработки"""
        if self.processing_log.exists():
            with open(self.processing_log) as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}

    def save_processing_log(self, data):
        """Сохраняет лог обработки"""
        with open(self.processing_log, "w") as f:
            json.dump(data, f, indent=2)

    def is_processed(self, book_path):
        """Проверяет, была ли книга обработана"""
        content_hash = self._get_content_hash(book_path)
        if not content_hash:
            return False
            
        log_data = self.load_processing_log()
        return content_hash in log_data.values()

    def archive_old_versions(self, book_path):
        """Архивирует старые версии книги"""
        current_hash = self._get_content_hash(book_path)
        if not current_hash:
            return
            
        log_data = self.load_processing_log()
        for archived_path, archived_hash in list(log_data.items()):
            if archived_hash == current_hash:
                try:
                    archived_file = Path(archived_path)
                    if archived_file.exists():
                        new_path = self.archive_path / archived_file.name
                        archived_file.rename(new_path)
                        print(f"📦 Архивировано: {archived_file.name}")
                        del log_data[archived_path]
                except Exception as e:
                    print(f"⚠️ Ошибка архивации: {str(e)}")
        
        self.save_processing_log(log_data)

    def mark_as_processed(self, book_path):
        """Помечает книгу как обработанную"""
        content_hash = self._get_content_hash(book_path)
        if not content_hash:
            return
            
        log_data = self.load_processing_log()
        log_data[str(book_path)] = content_hash
        self.save_processing_log(log_data)