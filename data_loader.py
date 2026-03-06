import string
import time
import os
import re
import ast
import html
from typing import List, Tuple, Optional, Dict, Any
import requests
import pandas as pd
import regex as re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)
from natasha import (
    Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
)

BASE_URL = 'https://lib.ru/HARMS/'
TARGET_SECTIONS = [
    "Стихи и рассказы",
    "Стихотворения",
    "Стихи для детей",
    "Рассказы для детей",
    "Проза, сценки, наброски",
    "Дневниковые записи",
    "Письма"
]
PUNCTUATION_TO_REMOVE = string.punctuation + '«»—'
RUSSIAN_STOPWORDS = set(stopwords.words("russian"))

# функция для удаления дубликатов -- иногда бывает так, что заголовки различаются, а тексты совпадают. 
def jaccard_similarity(lemmas_a: List[str], lemmas_b: List[str]) -> float:
    """Считает сходство Жаккара между двумя списками лемм."""
    if not lemmas_a or not lemmas_b:
        return 0.0
    set_a = set(lemmas_a)
    set_b = set(lemmas_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0

def deduplicate_results(results: List[Dict[str, Any]], lemmas_key: str = 'text_processed', threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Убирает дубликаты из списка результатов поиска по сходству лемм.

    Args:
        results:    список словарей с ключами title, text, text_processed, score
        lemmas_key: ключ, по которому лежит список лемм в словаре результата
        threshold:  порог сходства Жаккара (0..1).
    Returns:
        список уникальных результатов в исходном порядке
    """
    unique: List[Dict[str, Any]] = []
    seen_lemmas: List[List[str]] = []

    for item in results:
        lemmas = item.get(lemmas_key, [])
        if not isinstance(lemmas, list):
            lemmas = []

        is_duplicate = False
        for seen in seen_lemmas:
            if jaccard_similarity(lemmas, seen) >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(item)
            seen_lemmas.append(lemmas)

    removed = len(results) - len(unique)
    if removed > 0:
        print(f"удалено {removed} дублей")

    return unique

class WebScraper:
    """Загружает произведения с lib.ru."""

    def __init__(self, base_url: str = BASE_URL, timeout: int = 10):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def _decode_response(self, response: requests.Response) -> str:
        """Декодирует ответ из cp1251."""
        return response.content.decode('cp1251', errors='replace')

    def find_section_links(self, section_names: List[str]) -> List[str]:
        """Находит ссылки на разделы на главной странице."""
        try:
            response = self.session.get(self.base_url, timeout=self.timeout)
            response.raise_for_status()
            text = self._decode_response(response)

            soup = BeautifulSoup(text, 'html.parser')
            section_links = []

            for link_tag in soup.find_all('a', href=True):
                link_text = link_tag.get_text(strip=True)
                href = link_tag.get('href')

                for section in section_names:
                    if section.lower() in link_text.lower():
                        if href.startswith('http'):
                            full_url = href
                        else:
                            full_url = f'{self.base_url}{href}'
                        section_links.append(full_url)
                        break

            return section_links
        except:
            return []

    def scrape_section_html(self, section_urls: List[str]) -> str:
        """Загружает HTML контент со всех разделов."""
        all_html = ''
        for i, url in enumerate(section_urls, 1):
            try:
                response = self.session.get(url, timeout=self.timeout + 5)
                response.raise_for_status()
                text = self._decode_response(response)

                soup = BeautifulSoup(text, 'html.parser')
                for pre_tag in soup.find_all('pre'):
                    all_html += str(pre_tag)

                time.sleep(1)
            except:
                continue

        return all_html

    def extract_titles_and_texts(self, html_content: str) -> Tuple[List[str], List[str]]:
        """Извлекает названия и тексты из HTML."""
        text_parts = re.split(r'\<ul.*?ul\>', html_content, flags=re.DOTALL) # re.DOTALL -- любой символ включая \n
        title_parts = re.findall(r'\<ul.*?ul\>', html_content, flags=re.DOTALL)

        cleaned_titles = []
        for title_html in title_parts:
            title_text = re.sub(r'<.*?>', ' ', title_html)
            title_text = re.sub(r'\s+', ' ', title_text)
            cleaned = title_text.strip()
            if cleaned and len(cleaned) > 3:
                cleaned_titles.append(cleaned)

        cleaned_texts = []
        for text in text_parts[1:]:
            
            text = re.sub(r'<.*?>', ' ', text)
            text = html.unescape(text)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'^\d+\s+', '', text, flags=re.MULTILINE)
            cleaned = text.strip()
            if len(cleaned) > 10:
                cleaned_texts.append(cleaned)

        return cleaned_titles, cleaned_texts

    def scrape(self, target_sections: Optional[List[str]] = None) -> pd.DataFrame:
        """Основной метод для сбора данных."""
        sections = target_sections or TARGET_SECTIONS
        section_links = self.find_section_links(sections)

        if not section_links:
            try:
                direct_url = f'{self.base_url}harms.txt'
                response = self.session.get(direct_url, timeout=self.timeout + 5)
                response.raise_for_status()
                text = self._decode_response(response)

                parts = re.split(r'\n\s*\n', text)
                titles = [f"Часть {i+1}" for i in range(len(parts))]

                return pd.DataFrame({'title': titles, 'text': parts})
            except:
                return pd.DataFrame()

        all_html = self.scrape_section_html(section_links)
        if not all_html:
            return pd.DataFrame()

        titles, texts = self.extract_titles_and_texts(all_html)
        min_len = min(len(titles), len(texts))

        if min_len == 0:
            return pd.DataFrame()

        df = pd.DataFrame({
            'title': titles[:min_len],
            'text': texts[:min_len]
        })
        df = df.drop_duplicates(subset=['text'])
        df = df.reset_index(drop=True)

        return df

class TextPreprocessor:
    """Препооцессинг: лемматизация, удаление стоп-слов."""

    def __init__(self):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)

    def preprocess(self, text: str) -> List[str]:
        """Текст --> список лемм."""
        if not isinstance(text, str):
            text = str(text)

        cleaned = ''.join(
            char for char in text.lower()
            if char not in PUNCTUATION_TO_REMOVE
        )

        if not cleaned.strip():
            return []

        doc = Doc(cleaned)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        lemmas = []
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
            lemma = token.lemma
            if lemma and lemma not in RUSSIAN_STOPWORDS and len(lemma) > 1:
                lemmas.append(lemma)

        return lemmas

def split_into_sentences(text: str) -> List[str]:
    """Разбивает текст на предложения."""
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    cleaned_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 10:
            sent = re.sub(r'\s+', ' ', sent)
            cleaned_sentences.append(sent)

    return cleaned_sentences


def load_and_preprocess_texts(
    use_cached: bool = True,
    cache_file: str = 'harms_texts_processed.csv',
    split_sentences: bool = False,
    encoding: str = 'utf-8-sig'
) -> pd.DataFrame:
    """
    Загружает и обрабатывает тексты.

    Args:
        use_cached:      использовать кэшированные данные
        cache_file:      путь к кэш-файлу
        split_sentences: разбивать ли тексты на предложения
        encoding:        кодировка для сохранения CSV

    Returns:
        pd.DataFrame с колонками title, text, text_processed
    """
    if use_cached and os.path.exists(cache_file):
        df = None
        for enc in ['utf-8-sig', 'utf-8', 'cp1251']:
            try:
                df = pd.read_csv(cache_file, encoding=enc)
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            print("Не удалось прочитать кэш, загружаем заново...")
            return load_and_preprocess_texts(
                use_cached=False,
                cache_file=cache_file,
                split_sentences=split_sentences,
                encoding=encoding
            )

        # Восстанавливаем списки лемм из строк
        if 'text_processed' in df.columns:
            def safe_eval(x):
                try:
                    return ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
                except:
                    return []
            df['text_processed'] = df['text_processed'].apply(safe_eval)

        print(f"Загружено из кэша: {len(df)} документов")
        return df  
    # Загрузка с сайта
    print("Загрузка данных с сайта")
    scraper = WebScraper()
    df = scraper.scrape()

    if split_sentences:
        print("Разбивка на предложения")
        new_rows = []
        for idx, row in df.iterrows():
            sentences = split_into_sentences(row['text'])
            for i, sent in enumerate(sentences):
                new_rows.append({
                    'title': f"{row['title']} [предл. {i+1}]",
                    'text': sent
                })
        df = pd.DataFrame(new_rows)
        print(f"{len(df)} предложений")

    print("Лемматизация")
    preprocessor = TextPreprocessor()
    df['text_processed'] = df['text'].apply(preprocessor.preprocess)

    df = df[df['text_processed'].apply(len) > 0]
    df = df.reset_index(drop=True)

    df.to_csv(cache_file, index=False, encoding=encoding)

    return df

if __name__ == "__main__":
    df = load_and_preprocess_texts(use_cached=False, split_sentences=True, encoding='utf-8-sig')

    # if not df.empty:
    #     print(f"\nИтого: {len(df)} документов")
    #     print(f"\nПример текста: {df['text'].iloc[0][:100]}")
    #     print(f"Пример лемм:  {df['text_processed'].iloc[0][:10]}")
