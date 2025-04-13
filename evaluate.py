import time
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from bert_score import score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import nltk

from main import answer_question 

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_evaluation_csv(input_csv, output_csv):
    """
    1. Reads 'dataset.csv' containing columns [question, answer].
    2. For each row, calls answer_question() to get a generated answer.
    3. Writes out a new CSV with columns [question, expected, generated].
    """
    # Load the dataset
    df = pd.read_csv(input_csv)

    # Prepare a list to store results
    results = []

    # Iterate over each row in the dataset
    for idx, row in df.iterrows():
        question = row["question"]
        expected_answer = row["answer"]

        print(f"Generating answer for question #{idx+1}: {question}")

        # Call the RAG-based function from main.py
        generated_answer = answer_question(question)
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


def calculate_cosine_similarity(sentence1, sentence2):
    embeddings = model.encode([sentence1, sentence2])
    cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return cos_sim

def evaluate_answers(input_csv, output_scores_csv, output_averages_csv):
    df = pd.read_csv(input_csv)

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
    print("\nFinal Average Scores:\n", averages_df)

if __name__ == "__main__":
    # generate_evaluation_csv(
    # input_csv="dataset.csv",
    # output_csv="evaluation_results.csv"
    # )

    evaluate_answers(
        input_csv="evaluation_results.csv",
        output_scores_csv="final_evaluation_scores.csv",
        output_averages_csv="average_scores.csv"
    )
