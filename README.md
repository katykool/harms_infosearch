# Поисковая система по текстам Хармса
## Корпус
Тексты Даниила Хармса с сайта lib.ru. Корпус выбран из-за моей любви к Хармсу и из-за того, что мне хочется узнавать новые его стихотворения удобным способом. 
## Структура

```
data_loader.py      — загрузка и препроцессинг
matrix_indexing.py  — 6 индексов (TF-IDF и BM25)
search_engine.py    — обёртка над индексами, вывод результатов
main.py             — точка входа
```

## Установка

```bash
pip install requests pandas beautifulsoup4 nltk natasha regex scipy scikit-learn rank-bm25
```

## Использование

```bash
python main.py <запрос> [метод] [top_n]

python main.py "любовь"
python main.py "рыжий человек" bm25_matrix 10
```

## Методы поиска

| Метод | Описание |
|---|---|
| `tfidf_library` | TF-IDF через sklearn |
| `tfidf_manual` | TF-IDF через словари |
| `tfidf_matrix` | TF-IDF через матрицы |
| `bm25_library` | BM25 через rank_bm25 |
| `bm25_manual` | BM25 через словари |
| `bm25_matrix` | BM25 через матрицы *(по умолчанию)* |

