from PyPDF2 import PdfReader
from ebooklib import epub
from bs4 import BeautifulSoup
import logging

# Настройка логгирования
logging.basicConfig(filename='parser.log', level=logging.INFO)

def parse_pdf(file_path):
    """Извлечение текста из PDF"""
    try:
        text = ""
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""  # На случай пустой страницы
        return text
    except Exception as e:
        logging.error(f"PDF parsing failed: {file_path} - {str(e)}")
        raise

def parse_epub(file_path):
    """Извлечение текста из EPUB через ebooklib"""
    try:
        book = epub.read_epub(file_path)
        text = []
        for item in book.get_items():
            if item.get_type() == epub.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text.append(soup.get_text())
        return "\n".join(text)
    except Exception as e:
        logging.error(f"EPUB parsing failed: {file_path} - {str(e)}")
        raise

def parse_book(file_path):
    """Определяет формат книги и вызывает соответствующий парсер"""
    if str(file_path).lower().endswith('.pdf'):
        return parse_pdf(file_path)
    elif str(file_path).lower().endswith('.epub'):
        return parse_epub(file_path)
    else:
        raise ValueError(f"Unsupported format: {file_path}")