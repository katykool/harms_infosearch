"""
search_engine.py --- поисковая система.

6 версий индексов:
- TF-IDF: Library (sklearn), Manual (dict), Matrix (numpy/scipy)
- BM25: Library (rank_bm25), Manual (dict), Matrix (numpy)
"""

from typing import List, Dict, Tuple, Optional
import pandas as pd
from matrix_indexing import (
    TfidfIndexLibrary, TfidfIndexManual, TfidfIndexMatrix,
    Bm25IndexLibrary, Bm25IndexManual, Bm25IndexMatrix
)

class SearchEngine:
    """
    Методы поиска:
        search_tfidf_library: TF-IDF на sklearn
        search_tfidf_manual: TF-IDF на dict
        search_tfidf_matrix: TF-IDF на numpy
        search_bm25_library: BM25 на rank_bm25
        search_bm25_manual: BM25 на dict
        search_bm25_matrix: BM25 на numpy
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = 'text_processed',
        verbose: bool = True
    ):
        """
        Args:
            df (pd.DataFrame): таблица с текстами
            text_column (str): колонка с обработанными текстами (список лемм)
            verbose (bool): выводить ли информацию о построении индексов
        """
        self.df = df.copy()
        self.text_column = text_column
        self.verbose = verbose
        
        self.corpus = df[text_column].tolist()
        self.doc_titles = df['title'].tolist()
        self.doc_texts = df['text'].tolist()
        self.doc_count = len(self.corpus)
        
        self._tfidf_library = None
        self._tfidf_manual = None
        self._tfidf_matrix = None
        self._bm25_library = None
        self._bm25_manual = None
        self._bm25_matrix = None
        
        if self.verbose:
            print(f"Поисковая система инициализирована")
            print(f"Документов: {self.doc_count}")
 
    @property
    def tfidf_library(self):
        if self._tfidf_library is None:
            if self.verbose:
                print("\n[Построение TF-IDF индекса (sklearn)]")
            self._tfidf_library = TfidfIndexLibrary(
                self.corpus, 
                self.doc_titles
            )
        return self._tfidf_library
    
    @property
    def tfidf_manual(self):
        if self._tfidf_manual is None:
            if self.verbose:
                print("\n[Построение TF-IDF индекса (ручная реализация)]")
            self._tfidf_manual = TfidfIndexManual(
                self.corpus,
                self.doc_titles
            )
        return self._tfidf_manual
    
    @property
    def tfidf_matrix(self):
        if self._tfidf_matrix is None:
            if self.verbose:
                print("\n[Построение TF-IDF матрицы]")
            self._tfidf_matrix = TfidfIndexMatrix(
                self.corpus,
                self.doc_titles
            )
        return self._tfidf_matrix

    @property
    def bm25_library(self):
        if self._bm25_library is None:
            if self.verbose:
                print("\n[Построение BM25 индекса (rank_bm25)]")
            self._bm25_library = Bm25IndexLibrary(
                self.corpus,
                self.doc_titles
            )
        return self._bm25_library
    
    @property
    def bm25_manual(self):
        if self._bm25_manual is None:
            if self.verbose:
                print("\n[Построение BM25 индекса (ручная реализация)]")
            self._bm25_manual = Bm25IndexManual(
                self.corpus,
                self.doc_titles
            )
        return self._bm25_manual
    
    @property
    def bm25_matrix(self):
        if self._bm25_matrix is None:
            if self.verbose:
                print("\n[Построение BM25 матрицы]")
            self._bm25_matrix = Bm25IndexMatrix(
                self.corpus,
                self.doc_titles
            )
        return self._bm25_matrix
  
    def search_tfidf_library(self, query_tokens: List[str], top_n: int = 5) -> List[Tuple]:
        return self.tfidf_library.search(query_tokens, top_n)
    
    def search_tfidf_manual(self,query_tokens: List[str], top_n: int = 5) -> List[Tuple]:
        return self.tfidf_manual.search(query_tokens, top_n)
    
    def search_tfidf_matrix(self, query_tokens: List[str], top_n: int = 5) -> List[Tuple]:
        return self.tfidf_matrix.search(query_tokens, top_n)
    
    def search_bm25_library(self, query_tokens: List[str], top_n: int = 5) -> List[Tuple]:
        return self.bm25_library.search(query_tokens, top_n)
    
    def search_bm25_manual(self, query_tokens: List[str], top_n: int = 5) -> List[Tuple]:
        return self.bm25_manual.search(query_tokens, top_n)
    
    def search_bm25_matrix(self, query_tokens: List[str], top_n: int = 5) -> List[Tuple]:
        return self.bm25_matrix.search(query_tokens, top_n)

    def search(self, query_tokens: List[str], method: str = 'bm25_matrix', top_n: int = 5, verbose: bool = True) -> List[Tuple]:
        """
        Args:
            query_tokens (List[str]): Токены запроса (в леммах)
            method (str): Метод поиска
                - 'tfidf_library': TF-IDF (sklearn)
                - 'tfidf_manual': TF-IDF (ручная)
                - 'tfidf_matrix': TF-IDF (матричная)
                - 'bm25_library': BM25 (rank_bm25)
                - 'bm25_manual': BM25 (ручная)
                - 'bm25_matrix': BM25 (матричная) - ПО УМОЛЧАНИЮ
            top_n (int): Количество результатов
            verbose (bool): Выводить ли результаты

        Returns:
            List[Tuple]: Результаты поиска
        """
        search_methods = {
            'tfidf_library': self.search_tfidf_library,
            'tfidf_manual': self.search_tfidf_manual,
            'tfidf_matrix': self.search_tfidf_matrix,
            'bm25_library': self.search_bm25_library,
            'bm25_manual': self.search_bm25_manual,
            'bm25_matrix': self.search_bm25_matrix,
        }
        
        results = search_methods[method](query_tokens, top_n)
        
        if verbose:
            self._print_results(query_tokens, method, results)
        
        return results
    
    def _print_results(self, query_tokens: List[str], method: str, results: List[Tuple]) -> None:
        query_str = ' '.join(query_tokens)
        print(f"\nПоиск ({method}): '{query_str}'")
        
        if not results:
            print("Не найдено\n")
            return
        
        for rank, (doc_id, score, title) in enumerate(results, 1):
            preview = self.doc_texts[doc_id][:100]
            print(f"{rank}. [{score:.4f}] {title}")
            print(f"   {preview}...")
            print()

def create_search_engine(df: pd.DataFrame, text_column: str = 'text_processed') -> SearchEngine:
    """
    Создаёт поисковую систему из DataFrame.
    
    Args:
        df (pd.DataFrame): Таблица с текстами
        text_column (str): Колонка с лемами
    
    Returns:
        SearchEngine: Инициализированная поисковая система
    """
    return SearchEngine(df, text_column, verbose=True)

