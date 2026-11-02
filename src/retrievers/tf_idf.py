import re

from src.data_utils.document import Document
from src.retrievers.base_retriever import Retriever
import numpy as np
from scipy.sparse import lil_matrix


class TFIDF(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        super(TFIDF, self).__init__(documents, index_path)
        self.corpus_vocabulary = self.GetCorpusVocabulary()
        self.idf = self.GetInverseDocumentFrequencies()

        # Only run if we are in TFIDF class
        if type(self) == TFIDF:
            self.tfidf_matrix = self.GetDocumentsTFIDFVectors()

    def PreprocessText(self, text: str) -> list[str]:
        """
        Remove unwanted characters from text and lowercase
        """
        text = text.lower() # lowercase
        text = re.sub(r'[^a-z0-9\s]', ' ', text) # this can be modified
        return text.split()

    def GetQueryVocabulary(self, query: str) -> list[str]:
        """
        Returns all unique words in a query
        """
        return set(self.PreprocessText(query))

    def GetDocumentVocabulary(self, document: Document) -> list[str]:
        """
        Returns all unique words in a document
        """
        return set(self.PreprocessText(document.GetText()))

    def GetCorpusVocabulary(self) -> set[str]:
        """
        Returns all unique words in ALL documents.
        """
        # Use map function for more efficient processing
        processed_texts = map(lambda doc: set(self.PreprocessText(doc.GetText())), self.index.GetDocuments())
        corpus_vocabulary = set().union(*processed_texts)
        return corpus_vocabulary

    def GetDocumentFrequencies(self) -> dict[str, int]:
        """
        Returns a dict with terms and counts of how many documents the term occurs in (counted once per document).
        """
        dict_doc_freq = dict.fromkeys(self.GetCorpusVocabulary(), 0)
        for document in self.index.GetDocuments():
            unique_terms = self.GetDocumentVocabulary(document)
            for term in unique_terms:
                dict_doc_freq[term] += 1
        return dict_doc_freq

    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        """
        doc_counts = len(self.index.GetDocuments())
        dict_idf = {key: np.log(doc_counts / value) for key, value in self.GetDocumentFrequencies().items()}
        return dict_idf
    
    def GetDocumentTermCounts(self, document: Document):
        """
        Returns a dict on the form {term: num_occurences} for an input Document.
        """
        dict_term_counts = dict.fromkeys(self.GetDocumentVocabulary(document), 0)
        for term in self.PreprocessText(document.GetText()):
            dict_term_counts[term] += 1
        return dict_term_counts

    def GetDocumentsTFIDFVectors(self):
        """
        Returns a n_documents x n_terms matrix whose element (i, j) is the td-idf score of the j'th term in the i'th document.
        """
        # Create a mapping from terms to indices
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        # Initialize a sparse matrix
        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)
        tfidf_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)

        for doc_idx, document in enumerate(self.index.GetDocuments()):
            term_counts = self.GetDocumentTermCounts(document)
            if not term_counts:
                continue  # There were some errors due to empty documents in Fiqa
            max_freq = max(term_counts.values())  # Get the maximum based on the project description
            for term, count in term_counts.items():
                term_idx = self.term_to_index[term]
                tf_score = count / max_freq
                tfidf_matrix[doc_idx, term_idx] = tf_score * self.idf[term]
        
        return tfidf_matrix.tocsr()   # Convert to CSR format for efficient row slicing
    
    def QueryToVector(self, queries: list[str]):
        """ Converts a query or batch of queries into a sparse vector matching the vocab size with counts of how many times the term occurs.
        
        Args:
            query (str): The query to convert
        
        Returns:
            csr_matrix: A sparse vector representation of the query
        """
        query_vector = lil_matrix((len(queries), len(self.corpus_vocabulary)), dtype=np.float32)

        for idx, query in enumerate(queries):
            term_counts = dict.fromkeys(self.GetQueryVocabulary(query), 0)
            for term in self.PreprocessText(query):
                if term in self.term_to_index:
                    term_counts[term] += 1
                    term_idx = self.term_to_index[term]
                    query_vector[idx, term_idx] = term_counts[term]

        return query_vector.tocsr()

    #@time_func
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document

            num_queries x num_documents where (i, j) = ∑_i TF-IDF(q_i, D_j; D) 
        
        """
        # Convert the queries to a vector
        query_vectors = self.QueryToVector(queries) # n_queries x vocab size
        scores = query_vectors.dot(self.tfidf_matrix.T).toarray() # n_queries x n_documents
        return scores
    