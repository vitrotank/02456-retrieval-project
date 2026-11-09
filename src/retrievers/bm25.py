from scipy.sparse import lil_matrix

import re

from collections import Counter
from numpy import log as np_log


from scipy.sparse import lil_matrix,csr_matrix
from src.retrievers.tf_idf import TFIDF
from src.data_utils.document import Document





class BM25(TFIDF):
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75) -> None:
        super(BM25, self).__init__(documents, index_path)
        self.k1 = k1
        self.b = b

        self.documents = list(self.index.GetDocuments())

        self.document_term_counts = {document: Counter(self.PreprocessText(document.GetText())) for document in self.documents}

        self.document_lengths, self.average_document_length = self.GetDocumentLengths()

        self.corpus_vocabulary = self.GetCorpusVocabulary()

        self.term_to_index = {term: idx
                              for idx, term in enumerate(sorted(self.corpus_vocabulary))
                              }
        self.document_frequencies = self.GetDocumentFrequencies()

        self.inverse_document_frequencies = {term: self.InverseDocumentFrequency(term) for term in self.corpus_vocabulary}

        if type(self) == BM25:
            self.bm25_matrix = self.GetDocumentsBM25Vectors()

    def PreprocessText(self, text: str) -> list[str]:
        """
        Remove unwanted characters from text and lowercase
        """
        text = text.lower() # lowercase
        text = re.sub(r'[^a-z0-9\s]', ' ', text) # this can be modified
        return text.split()

    def GetQueryVocabulary(self, query: str) -> set[str]:
        """
        Returns all unique words in a query
        """
        return set(self.PreprocessText(query))

    def GetDocumentVocabulary(self, document: Document) -> set[str]:

        '''
        Return all unique terms in a document.
        '''
        return set(self.PreprocessText(document.GetText()))
    
    def GetDocumentLengths(self) -> tuple[dict[Document, int], float]:
        """
        Returns a dict with documents and their corresponding lengths, as well as the average document length.
        """
        document_lengths = {}
        average_document_length = 0

        for document in self.documents:
            document_lengths[document] =  len(self.PreprocessText(document.GetText()))

        if not document_lengths:
            return {}, 0
        average_document_length = (sum(document_lengths.values()) / len(document_lengths))

        return document_lengths, average_document_length

    ## Corpus - All documents #
    
    def GetCorpusVocabulary(self) -> set[str]:
         '''
         Returns all unique words in all documents.
         '''
         documents = list(self.index.GetDocuments())
         corpus_vocabulary = set()

         for document in documents:
              corpus_vocabulary.update(self.PreprocessText(document.GetText()))
              
         return corpus_vocabulary


    def GetDocumentTermCounts(self, document: Document) -> dict[str, int]:
        """
        Returns the term counts for one document
        """
        return self.document_term_counts[document]

    def GetDocumentFrequencies(self) -> dict[str, int]:
        '''
        Returns the number of documents containing each term.
        '''
        documents = list(self.index.GetDocuments())
        dict_doc_freq = dict.fromkeys(self.corpus_vocabulary,0)

        for document in documents:
            unique_terms = self.GetDocumentVocabulary(document)
            for term in unique_terms:
                dict_doc_freq[term] += 1

        return dict_doc_freq

    def TermFrequency(self, term: str, document: Document) -> float:
        
        frequency = self.document_term_counts[document].get(term, 0)

        if frequency == 0:
            return 0.0


        if self.average_document_length == 0:
            return 0.0

        document_length = self.document_lengths[document]

        length_normalization = (1- self.b + self.b * document_length/ self.average_document_length)

        numerator = frequency * (self.k1 + 1)

        denominator = (frequency+ self.k1 * length_normalization)

        return numerator / denominator

    def InverseDocumentFrequency(self, term: str) -> float:
        """
        Returns the BM25 inverse document frequency
        """
        number_of_documents = len(self.documents)
        document_frequency = self.document_frequencies.get(term, 0)
        if document_frequency == 0:
            return 0.0
        
        return np_log(1+ (number_of_documents- document_frequency+ 0.5)/ (document_frequency+ 0.5))

    def GetDocumentsBM25Vectors(self) -> csr_matrix:
        """
        Returns a sparse matrix containing BM25 document weights.
        """
    
        num_documents = len(self.documents)
        num_of_terms = len(self.corpus_vocabulary)
        bm25_matrix = lil_matrix((num_documents, num_of_terms))

        for doc_idx, document in enumerate(self.documents):
            term_counts = self.GetDocumentTermCounts(document)

            for term in term_counts:
                if term not in self.term_to_index:
                    continue

                term_idx = self.term_to_index[term]
                idf = self.inverse_document_frequencies[term]
                bm25_weight = self.TermFrequency(term, document) * idf
                bm25_matrix[doc_idx, term_idx] = bm25_weight

        return bm25_matrix.tocsr()

    def QueryToVector(self, queries: list[str]) -> csr_matrix:
        """
        Converts queries into sparse binary term vectors

        A value of 1 indicates that the terms occurs in the query
        IDF is alredy inclued in the BM25 matrix.

        """
        num_queries = len(queries)
        num_terms = len(self.corpus_vocabulary)
        query_matrix = lil_matrix((num_queries, num_terms))

        for query_idx, query in enumerate(queries):
            unique_query_terms = self.GetQueryVocabulary(query)
            for term in unique_query_terms:
                term_index = self.term_to_index.get(term)
                if term_index is not None:
                    query_matrix[query_idx, term_index] = 1.0
        return query_matrix.tocsr()
    
    def CalculateScores(self, queries: list[str]):
        """Calculate BM25 socres ofr every query-document pair

        Returns:

        np.darray :
        Matrix whtih shape (n_queries, n_documents)

        """


        

        query_matrix = self.QueryToVector(queries)
        scores = query_matrix.dot(self.bm25_matrix.T).toarray()
        return scores