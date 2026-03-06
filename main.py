import sys
from data_loader import load_and_preprocess_texts, TextPreprocessor, deduplicate_results
from search_engine import create_search_engine

def search_texts(
    query: str,
    method: str = 'bm25_matrix',
    top_n: int = 5,
    use_cached: bool = True,
    cache_file: str = 'harms_texts_processed.csv'
) -> None:
    """
    Args:
        query: Поисковый запрос (слова через пробел)
        method: 
            - 'tfidf_library': TF-IDF (sklearn)
            - 'tfidf_manual': TF-IDF (ручная через словари)
            - 'tfidf_matrix': TF-IDF (ручная через матрицы)
            - 'bm25_library': BM25 (rank_bm25)
            - 'bm25_manual': BM25 (ручная через словари)
            - 'bm25_matrix': BM25 (ручная через матрицы) [ПО УМОЛЧАНИЮ]
        top_n: Количество результатов
        use_cached: Использовать кэш
        cache_file: Путь к кэш-файлу
    """
    print("ПОИСК")
    
    # Загрузка, предобработка
    print("\n[1/2] Загрузка и предобработка")
    
    df = load_and_preprocess_texts(
        use_cached=use_cached,
        cache_file=cache_file
    )
    if df.empty:
        print("Ошибка при загрузке текстов")
        return
    
    # Поиск
    print("[2/2] Поиск")
    engine = create_search_engine(df)

    # Исправлено: лемматизируем запрос перед поиском
    preprocessor = TextPreprocessor()
    query_tokens = preprocessor.preprocess(query)

    print(f"\nЗапрос: '{query}'")
    print(f"Леммы запроса: {query_tokens}")
    print(f"Метод: {method}")

    if not query_tokens:
        print("Запрос пустой после лемматизации")
        return

    results = engine.search(
        query_tokens,
        method=method,
        top_n=top_n,
        verbose=True
    )
    
    if results:
        print(f"Найдено {len(results)} результатов")
    else:
        print("Не найдено")

def main() -> None:
    query = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'bm25_matrix'
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        
    search_texts(query, method=method, top_n=top_n)

if __name__ == "__main__":
    main()
