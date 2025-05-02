from app.database import Database
from app.book_parser import parse_book
from app.text_processor import TextProcessor
from app.trainer import NeurobitTrainer
from pathlib import Path
import time
import json

def initialize_environment():
    """Создает необходимые папки и файлы"""
    data_dir = Path("data")
    required_dirs = ["archive", "books", "model_weights"]
    
    for dir_name in required_dirs:
        (data_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    required_files = {
        "config.json": {
            "supported_langs": {
                "ru": {
                    "model": "ru_core_news_sm",
                    "slang_file": "slang_ru.json",
                    "vocab_file": "vocab_ru.txt"
                }
            }
        },
        "slang_ru.json": {},
        "vocab_ru.txt": ""
    }
    
    for filename, content in required_files.items():
        file_path = data_dir / filename
        if not file_path.exists():
            with open(file_path, "w", encoding="utf-8") as f:
                if isinstance(content, dict):
                    json.dump(content, f, ensure_ascii=False, indent=2)
                else:
                    f.write(content)

def main():
    # Инициализация окружения
    initialize_environment()
    
    # Инициализация компонентов
    db = Database()
    processor = TextProcessor()
    trainer = NeurobitTrainer()

    # Очистка устаревших записей
    removed = db.clean_orphaned_entries()
    print(f"Удалено устаревших записей: {removed}")

    # Обработка новых книг
    books = db.get_all_books()
    texts = []
    
    for book in books:
        if not db.is_processed(book):
            try:
                print(f"Обработка: {book.name}")
                text = parse_book(book)
                processed = processor.process_text(text)
                texts.append(processed)
                db.mark_as_processed(book)
                db.archive_old_versions(book)
            except Exception as e:
                print(f"Ошибка обработки {book.name}: {str(e)}")

    # Обучение модели
    if texts:
        print("Обновление словаря...")
        processor.build_vocabulary(texts)
        
        print("Начало обучения...")
        trainer.train_on_texts(texts)
        trainer.save_model()
        
        print("Обучение завершено. Модель сохранена.")
    else:
        print("Новых данных для обучения не найдено.")

if __name__ == "__main__":
    start_time = time.time()
    print("🚀 Запуск Neurobit...")
    main()
    print(f"⏱ Выполнено за: {time.time() - start_time:.2f} сек.")