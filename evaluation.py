import re
from collections import Counter
from opencc import OpenCC
cc = OpenCC('t2s')  # Converts Traditional to Simplified
import jieba
import spacy
from sklearn.metrics.pairwise import cosine_similarity

def normalize_text(text):
    """Lowercase, remove punctuation/articles/extra whitespace."""
    text = cc.convert(text)
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[^\w\u4e00-\u9fff]+', '', text)
    text = re.sub(r"[A-Za-z]", "", text)
    return text

def tokenize(text):
    return list(jieba.cut(text))

def spacy_score(prediction: str, ground_truth: str) -> float:
    nlp = spacy.load("zh_core_web_sm")
    prediction = normalize_text(prediction)
    ground_truth = normalize_text(ground_truth)
    prediction_tokens = " ".join(tokenize(prediction))
    ground_truth_tokens = " ".join(tokenize(ground_truth))
    gold_embedding = nlp(prediction_tokens).vector
    generated_embedding = nlp(ground_truth_tokens).vector
    return float(cosine_similarity([gold_embedding], [generated_embedding])[0][0])

def english_spacy_score(prediction: str, ground_truth: str) -> float:
    nlp = spacy.load("en_core_web_sm")
    prediction = normalize_text(prediction)
    ground_truth = normalize_text(ground_truth)
    prediction_tokens = " ".join(tokenize(prediction))
    ground_truth_tokens = " ".join(tokenize(ground_truth))
    gold_embedding = nlp(prediction_tokens).vector
    generated_embedding = nlp(ground_truth_tokens).vector
    return float(cosine_similarity([gold_embedding], [generated_embedding])[0][0])

def mrr_score(contexts, ground_truth: str) -> float:
    gt_tokens = set(tokenize(normalize_text(ground_truth)))
    # print(gt_tokens)
    maxs = 0
    score = 0.0
    for i in range(len(contexts)):
        context = contexts[i]
        context_tokens = set(tokenize(normalize_text(context)))
        common = gt_tokens & context_tokens
        if not gt_tokens:
            return 0.0
        # print(context_tokens)
        score = len(common) / len(gt_tokens)
        score = max(score, maxs)
    return score