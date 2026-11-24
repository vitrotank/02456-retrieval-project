import hydra
from src.data_utils.corpus import Corpus
from src.retrievers.tf_idf import TFIDF
from src.retrievers.bm25 import BM25
from src.data_utils.query import Query
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate
from torch.utils.data import DataLoader
from src.retrievers.dpr import DPR
from src.data_utils.dataset import DPRDataset
from src.training.trainer import Trainer

@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    ### Load a dataset
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')
    ### Prepare queries
    train_queries = [Query(corpus.train['question'][idx], 
                           id=corpus.train['query_id'][idx], 
                           relevant_document_ids=corpus.relevant_ids[corpus.train['query_id'][idx]]
                           ) 
                        for idx in range(corpus.train.num_rows)
                    ]

    val_queries =   [Query(corpus.val['question'][idx], 
                         id=corpus.val['query_id'][idx], 
                         relevant_document_ids=corpus.relevant_ids[corpus.val['query_id'][idx]]
                           ) 
                        for idx in range(corpus.val.num_rows)
                    ]
    
    test_queries =  [Query(corpus.test['question'][idx], 
                           id=corpus.test['query_id'][idx], 
                           relevant_document_ids=corpus.relevant_ids[corpus.test['query_id'][idx]]
                           ) 
                        for idx in range(corpus.test.num_rows)
                    ]
    
    ### Initialize TF-IDF class
    print("Creating TF-IDF...", flush=True)
    tf_idf = TFIDF(documents=corpus.all_documents)
    print("TF-IDF created.", flush=True)

    ### Initialize BM25 class
    print("Creating BM25...", flush=True)
    bm25 = BM25(documents=corpus.all_documents, k1=cfg.bm25.k1, b=cfg.bm25.b)
    print("BM25 created.", flush=True)
   
    
    ### Retrieve top-k relevant queries
    tfidf_train_relevancies = retrieveQueryAndGetRelevancies(tf_idf, train_queries, k=cfg.top_k)
    tfidf_val_relevancies = retrieveQueryAndGetRelevancies(tf_idf, val_queries, k=cfg.top_k)
    tfidf_test_relevancies = retrieveQueryAndGetRelevancies(tf_idf, test_queries, k=cfg.top_k)

    bm25_train_relevancies = retrieveQueryAndGetRelevancies(bm25, train_queries, k=cfg.top_k)
    bm25_val_relevancies = retrieveQueryAndGetRelevancies(bm25, val_queries, k=cfg.top_k)
    bm25_test_relevancies = retrieveQueryAndGetRelevancies(bm25, test_queries, k=cfg.top_k) 




    ### Evaluate
    train_scores = Evaluate(tfidf_train_relevancies, train_queries)
    val_scores = Evaluate(tfidf_val_relevancies, val_queries)
    test_scores = Evaluate(tfidf_test_relevancies, test_queries)

    bm25_train_scores = Evaluate(bm25_train_relevancies, train_queries)
    bm25_val_scores = Evaluate(bm25_val_relevancies, val_queries)
    bm25_test_scores = Evaluate(bm25_test_relevancies, test_queries)

    ### Train DPR
    print("Creating DPR...", flush=True)
    dpr = DPR(cfg)
    train_dataset = DPRDataset(corpus.train)
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    trainer = Trainer()
    trainer.train(dpr.query_encoder,
                  dpr.document_encoder,
                  train_loader,
                  dpr.tokenizer,
                  epochs=cfg.training.epochs,
                  lr=cfg.training.lr)
    print("DPR trained.", flush=True)

    ### Build DPR index with the trained weights
    print("Building DPR index...", flush=True)
    dpr.BuildIndex(corpus.all_documents)
    print("DPR index built.", flush=True)

    dpr_train_relevancies = retrieveQueryAndGetRelevancies(dpr, train_queries, k=cfg.top_k)
    dpr_val_relevancies = retrieveQueryAndGetRelevancies(dpr, val_queries, k=cfg.top_k)
    dpr_test_relevancies = retrieveQueryAndGetRelevancies(dpr, test_queries, k=cfg.top_k)

    dpr_train_scores = Evaluate(dpr_train_relevancies, train_queries)
    dpr_val_scores = Evaluate(dpr_val_relevancies, val_queries)
    dpr_test_scores = Evaluate(dpr_test_relevancies, test_queries)

    ### Print all results
    # Every metric only sees the top-k retrieved documents, so MRR is MRR@k
    k = cfg.top_k
    print(f'\nResults with cutoff k={k} (Recall@{k}, Precision@{k}, MRR@{k})')
    results = {
        'TF-IDF': (train_scores, val_scores, test_scores),
        'BM25': (bm25_train_scores, bm25_val_scores, bm25_test_scores),
        'DPR': (dpr_train_scores, dpr_val_scores, dpr_test_scores),
    }
    for method, splits in results.items():
        for split, scores in zip(['Train', 'Val', 'Test'], splits):
            print(f'{method:<7}{split:<6} Recall@{k}={scores["recall"]:.4f}  '
                  f'Precision@{k}={scores["precision"]:.4f}  MRR@{k}={scores["mrr"]:.4f}')

if __name__ == '__main__':
    main()