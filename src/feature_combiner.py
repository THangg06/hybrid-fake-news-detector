import numpy as np

def combine_features(tfidf_matrix, roberta_embeddings, roberta_weight=2.0):
    """
    Concatenate text features (TF-IDF + RoBERTa + optional engineered stats).
    
    Args:
        tfidf_matrix: TF-IDF features (already normalized 0-1)
        roberta_embeddings: RoBERTa embeddings (scaled to emphasize)
        engineered_features: Optional handcrafted features from raw content
        roberta_weight: Multiplier for RoBERTa embeddings (default 2.0)
    
    Returns:
        Combined feature matrix
    """

    feature_blocks = [
        tfidf_matrix,           # TF-IDF features from text
        roberta_embeddings,         # RoBERTa embeddings from text
    ]

    return np.hstack(tuple(feature_blocks))
    
# Total: TF-IDF dims + 768 RoBERTa dims + optional engineered dims
