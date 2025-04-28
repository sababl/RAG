"""
Evaluation module for RAG system performance.

This module provides utilities for evaluating the quality of RAG-generated answers
against expected answers using various NLP metrics.
"""

import time
from typing import Dict, List, Any, Tuple

import pandas as pd
import nltk
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from bert_score import score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from main import answer_question
from config import MODEL_NAME

# Download required NLTK resources
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize embedding model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_evaluation_csv(input_csv: str, output_csv: str) -> None:
    """
    Generate evaluation data by running RAG on a set of questions.

    Args:
        input_csv (str): Path to input CSV with 'question' and 'answer' columns
        output_csv (str): Path to output CSV where results will be stored
    
    Raises:
        FileNotFoundError: If input_csv doesn't exist
        KeyError: If input_csv doesn't have required columns
    """
    # Load the dataset
    df = pd.read_csv(input_csv)

    if 'question' not in df.columns or 'answer' not in df.columns:
        raise KeyError("Input CSV must contain 'question' and 'answer' columns")

    # Prepare a list to store results
    results = []

    # Iterate over each row in the dataset
    for idx, row in df.iterrows():
        question = row["question"]
        expected_answer = row["answer"]

        print(f"Generating answer for question #{idx+1}: {question}")

        # Call the RAG-based function from main.py
        generated_answer, _ = answer_question(question)
        time.sleep(5)  # Add a delay to avoid rate limits
        results.append({
            "question": question,
            "expected": expected_answer,
            "generated": generated_answer
        })

    # Create a new DataFrame with the results
    results_df = pd.DataFrame(results)

    # Save to CSV
    results_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\nCreated CSV with {len(results_df)} rows: {output_csv}")


def calculate_cosine_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculate cosine similarity between two sentences using embeddings.
    
    Args:
        sentence1 (str): First sentence
        sentence2 (str): Second sentence
        
    Returns:
        float: Cosine similarity score between 0 and 1
    """
    embeddings = model.encode([sentence1, sentence2])
    cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return cos_sim


def evaluate_answers(input_csv: str, output_scores_csv: str, output_averages_csv: str) -> Dict[str, float]:
    """
    Evaluate the quality of generated answers using multiple metrics.
    
    Args:
        input_csv (str): Path to CSV with questions, expected and generated answers
        output_scores_csv (str): Path to save per-item scores
        output_averages_csv (str): Path to save average scores
        
    Returns:
        Dict[str, float]: Dictionary of average scores for each metric
        
    Raises:
        FileNotFoundError: If input_csv doesn't exist
    """
    df = pd.read_csv(input_csv)
    
    if any(col not in df.columns for col in ['question', 'expected', 'generated']):
        raise KeyError("Input CSV must contain 'question', 'expected', and 'generated' columns")

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        question = row['question']
        expected = row['expected']
        generated = row['generated']

        # BLEU Score
        bleu = sentence_bleu([expected.split()], generated.split())

        # ROUGE-L Score
        rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rougeL = rouge.score(expected, generated)['rougeL'].fmeasure

        # METEOR Score
        meteor = meteor_score(
            [word_tokenize(expected)], 
            word_tokenize(generated)
        )

        # BERTScore
        _, _, F1 = score(
            [generated], 
            [expected], 
            lang='en', 
            model_type='bert-base-uncased'
        )
        bertscore_f1 = F1.mean().item()

        # Cosine Similarity
        cosine_sim = calculate_cosine_similarity(generated, expected)

        results.append({
            'question': question,
            'expected': expected,
            'generated': generated,
            'BLEU': bleu,
            'ROUGE-L': rougeL,
            'METEOR': meteor,
            'BERTScore_F1': bertscore_f1,
            'Cosine_Similarity': cosine_sim
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_scores_csv, index=False)
    print(f"Per-item evaluation completed. Scores saved to {output_scores_csv}")

    metrics = ['BLEU', 'ROUGE-L', 'METEOR', 'BERTScore_F1', 'Cosine_Similarity']
    averages = {metric: results_df[metric].mean() for metric in metrics}

    averages_df = pd.DataFrame([averages])
    averages_df.to_csv(output_averages_csv, index=False)
    print(f"Average scores saved to {output_averages_csv}")
    print(f"\nFinal Average Scores for model {MODEL_NAME}:\n", averages_df)
    
    return averages


if __name__ == "__main__":
    generate_evaluation_csv(
        input_csv="dataset.csv",
        output_csv="results/evaluation_results.csv"
    )

    evaluate_answers(
        input_csv="results/evaluation_results.csv",
        output_scores_csv="results/final_evaluation_scores.csv",
        output_averages_csv="results/average_scores.csv"
    )
