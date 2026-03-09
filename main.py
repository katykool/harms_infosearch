import sys
from data_loader import load_and_preprocess_texts, TextPreprocessor
from search_engine import create_search_engine

def search_texts(query, method = 'bm25_matrix', top_n = 5, use_cached = True, cache_file = 'harms_texts_processed.csv'):
    print("ПОИСК")
    
    # Загрузка и предобработка
    print("\n[1/2] Загрузка и предобработка")
    
    df = load_and_preprocess_texts(use_cached=use_cached, cache_file=cache_file)    
    
    # Поиск
    print("[2/2] Поиск")
    engine = create_search_engine(df)

    # лемматизируем запрос перед поиском
    preprocessor = TextPreprocessor()
    query_tokens = preprocessor.preprocess(query)

    engine.search(query_tokens, method=method, top_n=top_n)


def main():
    query = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'bm25_matrix'
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    search_texts(query, method=method, top_n=top_n)

if __name__ == "__main__":
    main()
