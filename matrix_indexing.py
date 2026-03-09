import math
from collections import defaultdict, Counter
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# реализация с помощью библиотеки
class TfidfIndexLibrary:    
    def __init__(self, processed_texts, doc_titles):       
        self.doc_titles = doc_titles
        self.doc_count = len(processed_texts)
        text_strings = [' '.join(lemmas) for lemmas in processed_texts] # sklearn требует строки лемм, а не списки
        self.vectorizer = TfidfVectorizer(token_pattern=None, tokenizer=lambda x: x.split(' ')) # tokenizer -- делим просто по пробелам
        self.tfidf_matrix = self.vectorizer.fit_transform(text_strings)
    
    def search(self, query_tokens, top_n = 5):
        query_string = ' '.join(query_tokens)
        query_vec = self.vectorizer.transform([query_string])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_n]
        return [(idx, float(similarities[idx]), self.doc_titles[idx]) for idx in top_indices if similarities[idx] > 0]

# реализация через словари
class TfidfIndexManual:
    def __init__(self, processed_texts, doc_titles): # создаем словари для подсчёта
        self.doc_titles = doc_titles
        self.doc_count = len(processed_texts)
        self.freq_index = defaultdict(dict)
        self.doc_freq = defaultdict(int)
        self.doc_lengths = []
        for doc_id, tokens in enumerate(processed_texts):
            self.doc_lengths.append(len(tokens))
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self.freq_index[term][doc_id] = count
            for term in set(tokens):
                self.doc_freq[term] += 1
            
    def search(self, query_tokens, top_n=5):
        relevant_docs = set()
        for token in query_tokens: # документы, которые содержат хотя бы один токен из запроса
            if token in self.freq_index:
                relevant_docs.update(self.freq_index[token].keys())
        scores = {} 
        for doc_id in relevant_docs: 
            score = 0
            doc_len = self.doc_lengths[doc_id]
            for token in query_tokens:
                if token in self.freq_index and doc_id in self.freq_index[token]:
                    tf = self.freq_index[token][doc_id] / doc_len if doc_len > 0 else 0 
                    idf = math.log(self.doc_count / (self.doc_freq[token] + 1))
                    score += tf * idf
            scores[doc_id] = score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [(doc_id, score, self.doc_titles[doc_id])for doc_id, score in sorted_results if score > 0]

# матричная реализация
class TfidfIndexMatrix:
    def __init__(self, processed_texts, doc_titles):
        self.doc_titles = doc_titles
        self.doc_count = len(processed_texts)
        
        # словарь
        vocab = set()
        for tokens in processed_texts:
            vocab.update(tokens)
        self.vocab = sorted(vocab)
        self.term_to_idx = {term: i for i, term in enumerate(self.vocab)}
        vocab_size = len(self.vocab)
        
        # IDF
        rows, cols, data = [], [], []
        idf_values = np.zeros(vocab_size)
        
        for term_id, term in enumerate(self.vocab):
            doc_count_with_term = sum(1 for tokens in processed_texts if term in tokens)
            idf_values[term_id] = math.log((self.doc_count + 1) / (doc_count_with_term + 1))
        
        # TF-IDF
        for doc_id, tokens in enumerate(processed_texts):
            if not tokens:  # пустые документы
                continue
                
            doc_len = len(tokens)
            term_counts = Counter(tokens)
            
            for term, count in term_counts.items():
                term_id = self.term_to_idx[term]
                tf = count / doc_len
                tfidf_value = tf * idf_values[term_id]
                
                rows.append(doc_id)
                cols.append(term_id)
                data.append(tfidf_value)
        
        self.tfidf_matrix = csr_matrix((data, (rows, cols)), shape=(self.doc_count, vocab_size))
    
    def search(self, query_tokens, top_n=5):
        # проверка на пустой запрос
        if not query_tokens:
            return []
        
        # индексы слов из запроса
        query_indices = []
        for token in query_tokens:
            if token in self.term_to_idx:
                query_indices.append(self.term_to_idx[token])
        
        if not query_indices:  
            return []
        
        # вектор запроса
        query_data = [1.0] * len(query_indices)
        query_norm = np.sqrt(len(query_indices))
        if query_norm > 0:
            query_data = [x / query_norm for x in query_data]
        
        rows = np.zeros(len(query_data))
        query_vec = csr_matrix((query_data, (rows, query_indices)), shape=(1, self.tfidf_matrix.shape[1]))
        
        # схожесть
        similarities = self.tfidf_matrix.dot(query_vec.T).toarray().flatten()
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        # результаты
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((int(idx), float(similarities[idx]), self.doc_titles[idx]))
        
        return results
    
# реалтзация с библиотекой
class Bm25IndexLibrary:    
    def __init__(self, processed_texts, doc_titles):
        self.doc_titles = doc_titles
        self.bm25 = BM25Okapi(processed_texts)
    
    def search(self, query_tokens, top_n=5):
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [(int(idx), float(scores[idx]), self.doc_titles[int(idx)]) for idx in top_indices if scores[idx] > 0]

# реализация через словари
class Bm25IndexManual:
    def __init__(self, processed_texts, doc_titles, k1 = 1.5, b = 0.75):
        self.doc_titles = doc_titles
        self.doc_count = len(processed_texts)
        self.k1 = k1
        self.b = b
        self.freq_index = defaultdict(dict)
        self.doc_freq = defaultdict(int)
        self.doc_lengths = []
        
        for doc_id, tokens in enumerate(processed_texts):
            self.doc_lengths.append(len(tokens))
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self.freq_index[term][doc_id] = count
            for term in set(tokens):
                self.doc_freq[term] += 1
        
        self.avg_doc_length = np.mean(self.doc_lengths) if self.doc_lengths else 1
        self.idf_values = {}
        for term in self.freq_index:
            self.idf_values[term] = math.log(
                (self.doc_count - self.doc_freq[term] + 0.5) /
                (self.doc_freq[term] + 0.5) + 1
            )
            
    def search(self, query_tokens, top_n=5):
        relevant_docs = set()
        for token in query_tokens:
            if token in self.freq_index:
                relevant_docs.update(self.freq_index[token].keys())
       
        scores = {}
        for doc_id in relevant_docs:
            score = 0
            doc_len = self.doc_lengths[doc_id]
            
            for token in query_tokens:
                if token in self.freq_index and doc_id in self.freq_index[token]:
                    tf = self.freq_index[token][doc_id]
                    idf = self.idf_values[token]
                    
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (
                        1 - self.b + self.b * doc_len / self.avg_doc_length
                    )
                    if denominator != 0:
                        score += idf * (numerator / denominator)
            scores[doc_id] = score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return [(doc_id, score, self.doc_titles[doc_id]) for doc_id, score in sorted_results if score > 0]


class Bm25IndexMatrix:
    def __init__(self, processed_texts, doc_titles, k1 = 1.5, b = 0.75):
        self.doc_titles = doc_titles
        self.doc_count = len(processed_texts)
        self.k1 = k1
        self.b = b
                
        vocab = set()
        for tokens in processed_texts:
            vocab.update(tokens)
        
        self.vocab = sorted(list(vocab))
        self.term_to_idx = {term: i for i, term in enumerate(self.vocab)}
        vocab_size = len(self.vocab)
        
        rows, cols, data = [], [], []
        self.doc_lengths = np.zeros(self.doc_count)
        
        for doc_id, tokens in enumerate(processed_texts):
            self.doc_lengths[doc_id] = len(tokens)
            term_counts = Counter(tokens)
            
            for term, count in term_counts.items():
                term_id = self.term_to_idx[term]
                rows.append(doc_id)
                cols.append(term_id)
                data.append(count)
        
        self.term_matrix = csr_matrix((data, (rows, cols)), shape=(self.doc_count, vocab_size), dtype=np.float32)
        self.avg_doc_length = np.mean(self.doc_lengths)
        doc_freq = np.asarray(self.term_matrix.sum(axis=0)).flatten()
        self.idf_values = np.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
    
    def search(self, query_tokens, top_n = 5):
        query_indices = []
        for token in query_tokens:
            if token in self.term_to_idx:
                query_indices.append(self.term_to_idx[token])
        
        scores = np.zeros(self.doc_count)
        
        for term_id in query_indices:
            term_freqs = np.asarray(self.term_matrix[:, term_id].todense()).flatten()
            idf = self.idf_values[term_id]
            numerator = term_freqs * (self.k1 + 1)
            denominator = (term_freqs + self.k1 * (1 - self.b + self.b * self.doc_lengths / self.avg_doc_length))
            scores += idf * (numerator / denominator)
        top_indices = np.argsort(scores)[::-1][:top_n]
        
        return [(int(idx), float(scores[idx]), self.doc_titles[int(idx)]) for idx in top_indices if scores[idx] > 0]
