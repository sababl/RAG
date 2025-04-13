import requests
import time

import html
import pandas as pd

from bs4 import BeautifulSoup
import os


def clean_html(raw_html):
    soup = BeautifulSoup(html.unescape(raw_html), "html.parser")
    return soup.get_text(separator="\n")


def fetch_questions(page=1):
    url = "https://api.stackexchange.com/2.3/questions"
    params = {
        "order": "desc",
        "sort": "votes",
        "tagged": "python",
        "site": "stackoverflow",
        "pagesize": 20,
        "page": page
    }

    response = requests.get(url, params=params)
    data = response.json()
    return data.get("items", [])

def fetch_answer(answer_id):
    url = f"https://api.stackexchange.com/2.3/answers/{answer_id}"
    params = {
        "order": "desc",
        "sort": "activity",
        "site": "stackoverflow",
        "filter": "withbody"
    }

    response = requests.get(url, params=params)
    items = response.json().get("items", [])
    if items:
        return clean_html(items[0]["body"])
    return None
def save_to_csv(df, filename="dataset.csv"):
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"\n Saved {len(df)} entries to {filename}")

def main():
    page_max = 10
    dataset = pd.DataFrame(columns=["question", "answer"])

    for page in range(1, page_max):
        questions = fetch_questions(page)
        if not questions:
            print("No more questions found.")
            break
        for q in questions:
            title = clean_html(q.get("title", ""))
            body = clean_html(q.get("body", ""))
            accepted_answer_id = q.get("accepted_answer_id")

            if not accepted_answer_id:
                continue

            print(f"Fetching answer for: {title}")
            answer = fetch_answer(accepted_answer_id)
            time.sleep(0.2)  # Respect API rate limits
            if answer:
                new_row = {"question": title, "answer": answer}
                dataset = pd.concat([dataset, pd.DataFrame([new_row])], ignore_index=True)    
    if not dataset.empty:
        if os.path.exists("dataset.csv"):
            os.remove("dataset.csv")
        save_to_csv(dataset)
        print(f"\n Saved {len(dataset)} entries to dataset.csv")
    else:
        print("No clean Python Q&A found.")

if __name__ == "__main__":
    main()