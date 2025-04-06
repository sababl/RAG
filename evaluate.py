
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

# Ensure NLTK resources are available
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

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
        question = row['question']
        expected = row['expected']
        generated = row['generated']

        # BLEU Score
        bleu = sentence_bleu([expected.split()], generated.split())

        # ROUGE-L Score
        rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rougeL = rouge.score(expected, generated)['rougeL'].fmeasure

        # METEOR Score
        meteor = meteor_score([word_tokenize(expected)], word_tokenize(generated))

        # BERTScore
        _, _, F1 = score([generated], [expected], lang='en', model_type='bert-base-uncased')
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
    results_df.to_csv(output_csv, index=False)
    print(f"Evaluation completed. Results saved to {output_csv}")

# Run evaluation
if __name__ == "__main__":
    evaluate_answers("evaluation_results.csv", "final_evaluation_scores.csv")
