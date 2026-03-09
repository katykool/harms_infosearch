from matrix_indexing import (
    TfidfIndexLibrary, TfidfIndexManual, TfidfIndexMatrix,
    Bm25IndexLibrary, Bm25IndexManual, Bm25IndexMatrix
)

class SearchEngine:
    def __init__(self, df, text_column = 'text_processed'):
        self.df = df.copy()
        self.text_column = text_column
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
 
    @property
    def tfidf_library(self):
        if self._tfidf_library is None:
            self._tfidf_library = TfidfIndexLibrary(self.corpus, self.doc_titles)
        return self._tfidf_library
    
    @property
    def tfidf_manual(self):
        if self._tfidf_manual is None:
            self._tfidf_manual = TfidfIndexManual(self.corpus, self.doc_titles)
        return self._tfidf_manual
    
    @property
    def tfidf_matrix(self):
        if self._tfidf_matrix is None:
            self._tfidf_matrix = TfidfIndexMatrix(self.corpus, self.doc_titles)
        return self._tfidf_matrix

    @property
    def bm25_library(self):
        if self._bm25_library is None:
            self._bm25_library = Bm25IndexLibrary(self.corpus, self.doc_titles)
        return self._bm25_library
    
    @property
    def bm25_manual(self):
        if self._bm25_manual is None:
            self._bm25_manual = Bm25IndexManual(self.corpus, self.doc_titles)
        return self._bm25_manual
    
    @property
    def bm25_matrix(self):
        if self._bm25_matrix is None:
            self._bm25_matrix = Bm25IndexMatrix(self.corpus, self.doc_titles)
        return self._bm25_matrix
  
    def search_tfidf_library(self, query_tokens, top_n = 5):
        return self.tfidf_library.search(query_tokens, top_n)
    
    def search_tfidf_manual(self,query_tokens, top_n = 5):
        return self.tfidf_manual.search(query_tokens, top_n)
    
    def search_tfidf_matrix(self, query_tokens, top_n = 5):
        return self.tfidf_matrix.search(query_tokens, top_n)
    
    def search_bm25_library(self, query_tokens, top_n = 5):
        return self.bm25_library.search(query_tokens, top_n)
    
    def search_bm25_manual(self, query_tokens, top_n = 5):
        return self.bm25_manual.search(query_tokens, top_n)
    
    def search_bm25_matrix(self, query_tokens, top_n = 5):
        return self.bm25_matrix.search(query_tokens, top_n)

    def search(self, query_tokens, method = 'bm25_matrix', top_n = 5):
        search_methods = {
            'tfidf_library': self.search_tfidf_library,
            'tfidf_manual': self.search_tfidf_manual,
            'tfidf_matrix': self.search_tfidf_matrix,
            'bm25_library': self.search_bm25_library,
            'bm25_manual': self.search_bm25_manual,
            'bm25_matrix': self.search_bm25_matrix,
        }
        results = search_methods[method](query_tokens, top_n)
        
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


def create_search_engine(df, text_column = 'text_processed'):
    return SearchEngine(df, text_column)

