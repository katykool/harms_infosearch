import string
import time
import re
import ast
import html
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

all_url = 'https://lib.ru/HARMS/'
all_sections = ["Стихи и рассказы", "Стихотворения",  "Стихи для детей", "Рассказы для детей", "Проза, сценки, наброски", "Дневниковые записи", "Письма"]
punct = string.punctuation + '«»—'
sw = set(stopwords.words("russian"))

# функция для удаления дубликатов -- иногда бывает так, что заголовки различаются, а тексты совпадают. 
# функция для подсчёта сходства текстов для удаления
def jaccard_similarity(lemmas_a, lemmas_b):
    if not lemmas_a or not lemmas_b:
        return 0.0
    set_a = set(lemmas_a)
    set_b = set(lemmas_b)
    return len(set_a & set_b) / len(set_a | set_b) if len(set_a | set_b) > 0 else 0.0

def deduplicate_results(results, lemmas_key='text_processed', threshold=0.7):
    unique = []
    seen_lemmas = []

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

    # removed = len(results) - len(unique)
    # if removed > 0:
        # print(f"удалено {removed} дублей")

    return unique

class WebScraper:
    def __init__(self, base_url=all_url, timeout = 10):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

    def _decode_response(self, response: requests.Response):
        return response.content.decode('cp1251', errors='replace')

    def find_section_links(self, section_names): 
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

    def scrape_section_html(self, section_urls):
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

    def extract_titles_and_texts(self, html_content):
        text_parts = re.split(r'\<ul.*?ul\>', html_content, flags=re.DOTALL) # re.DOTALL -- любой символ включая \n
        title_parts = re.findall(r'\<ul.*?ul\>', html_content, flags=re.DOTALL)

        cleaned_titles = []
        for title_html in title_parts:
            title_text = re.sub(r'<.*?>', ' ', title_html) # отделение заголовков
            title_text = re.sub(r'\s+', ' ', title_text)
            cleaned = title_text.strip()
            if cleaned and len(cleaned) > 3:
                cleaned_titles.append(cleaned)

        cleaned_texts = []
        for text in text_parts[1:]:
            
            text = re.sub(r'<.*?>', ' ', text) # выделение текста
            text = html.unescape(text)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'^\d+\s+', '', text, flags=re.MULTILINE)
            cleaned = text.strip()
            if len(cleaned) > 10:
                cleaned_texts.append(cleaned)

        return cleaned_titles, cleaned_texts

    def scrape(self, target_sections=all_sections): # основная функция для парсинга
        sections = target_sections
        section_links = self.find_section_links(sections)

        if not section_links:
            direct_url = f'{self.base_url}harms.txt'
            response = self.session.get(direct_url, timeout=self.timeout + 5)
            text = self._decode_response(response)

            parts = re.split(r'\n\s*\n', text)
            titles = [f"Часть {i+1}" for i in range(len(parts))]

            return pd.DataFrame({'title': titles, 'text': parts})

        all_html = self.scrape_section_html(section_links)
        titles, texts = self.extract_titles_and_texts(all_html)
        min_len = min(len(titles), len(texts))

        df = pd.DataFrame({
            'title': titles[:min_len],
            'text': texts[:min_len]
        })
        df = df.drop_duplicates(subset=['text'])
        df = df.reset_index(drop=True)
        return df

class TextPreprocessor:
    def __init__(self):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)

    def preprocess(self, text):
        if not isinstance(text, str):
            text = str(text)
        cleaned = ''.join(char for char in text.lower() if char not in punct)
        doc = Doc(cleaned)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        lemmas = []
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
            lemma = token.lemma
            if lemma not in sw and len(lemma) > 1:
                lemmas.append(lemma)
        return lemmas

def load_and_preprocess_texts(use_cached = True, cache_file = 'harms_texts_processed.csv', encoding = 'utf-8-sig'):
    if use_cached:
        df = None
        for enc in ['utf-8-sig', 'utf-8', 'cp1251']:
            try:
                df = pd.read_csv(cache_file, encoding=enc)
                break
            except UnicodeDecodeError:
                continue

        if df is None: # загружаем данные с сайта заново
            return load_and_preprocess_texts(
                use_cached=False,
                cache_file=cache_file,
                encoding=encoding
            )

        # восстанавливаем списки лемм из строк
        def safe_eval(x):
            return ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
        df['text_processed'] = df['text_processed'].apply(safe_eval)

        # print(f"Загружено из кэша: {len(df)} документов")
        return df  
    
    # Загрузка с сайта
    # print("Загрузка данных с сайта")
    scraper = WebScraper()
    df = scraper.scrape()

    # print("Лемматизация")
    preprocessor = TextPreprocessor() # препроцессинг
    df['text_processed'] = df['text'].apply(preprocessor.preprocess)
    df = df.reset_index(drop=True)
    df.to_csv(cache_file, index=False, encoding=encoding)

    return df

if __name__ == "__main__":
    df = load_and_preprocess_texts(use_cached=False, encoding='utf-8-sig')

    # if not df.empty:
    #     print(f"\nИтого: {len(df)} документов")
    #     print(f"\nПример текста: {df['text'].iloc[0][:100]}")
    #     print(f"Пример лемм:  {df['text_processed'].iloc[0][:10]}")
