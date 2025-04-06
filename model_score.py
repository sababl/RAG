import pandas as pd
from bert_score import score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load pre-trained sentence transformer model for cosine similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to calculate cosine similarity
def calculate_cosine_similarity(sentence1, sentence2):
    embeddings = model.encode([sentence1, sentence2])
    cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return cos_sim

# Evaluation script
def evaluate_answers(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        expected = row['expected']
        generated = row['generated']

        # Calculate BERTScore
        P, R, F1 = score([generated], [expected], lang='en', model_type='bert-base-uncased')

        # Calculate Cosine Similarity
        cosine_sim = calculate_cosine_similarity(generated, expected)

        results.append({
            'question': row['question'],
            'expected': expected,
            'generated': generated,
            'BERTScore_F1': F1.mean().item(),
            'Cosine_Similarity': cosine_sim
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Evaluation completed. Results saved to {output_csv}")

# Run evaluation
if __name__ == "__main__":
    evaluate_answers("evaluation_results.csv", "bert_evaluation_scores.csv")
