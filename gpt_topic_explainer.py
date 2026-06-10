import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

MAX_WORKERS = 3
REQUEST_TIMEOUT = 120


def generate_topic_label(topic_id, keywords):
    prompt = f"""
You are generating a topic label for a research cluster.

Use ONLY the information from the keywords below.
Do NOT introduce new concepts or assumptions.
If the keywords are broad or mixed, generate a general label.

Keywords: {keywords}

Rules:
- Use 2–5 words only
- Do NOT add words that are not clearly supported by the keywords
- Prefer general terms over specific ones
- Avoid words like "abnormal", "anomaly", "traffic" unless explicitly present

Return ONLY the topic label.
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        return {
            "Topic": topic_id,
            "Keywords": keywords,
            "Topic_Label": data.get("response", "").strip()
        }

    except Exception as e:
        print(f"⚠️ Error generating label for Topic {topic_id}: {e}")
        return {
            "Topic": topic_id,
            "Keywords": keywords,
            "Topic_Label": "Error"
        }


def run_generatetopic(
    input_file="data/processed/bertopic/topic_info.csv",
    output_file="data/processed/bertopic/topic_labels.csv"
):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    topic_info = pd.read_csv(input_file)
    topic_info = topic_info[topic_info["Topic"] != -1].copy()

    tasks = []
    for _, row in topic_info.iterrows():
        tasks.append((row["Topic"], row["Representation"]))

    results = []

    print(f"🚀 Generating topic labels in parallel with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(generate_topic_label, topic_id, keywords): topic_id
            for topic_id, keywords in tasks
        }

        for future in as_completed(future_map):
            topic_id = future_map[future]
            try:
                result = future.result()
                results.append(result)
                print(f"✅ Saved Topic {topic_id}")
            except Exception as e:
                print(f"⚠️ Failed Topic {topic_id}: {e}")
                results.append({
                    "Topic": topic_id,
                    "Keywords": "",
                    "Topic_Label": "Error"
                })

    results = sorted(results, key=lambda x: x["Topic"])

    df_labels = pd.DataFrame(results)
    df_labels.to_csv(output_file, index=False)

    print(f"✅ topic_labels.csv saved to: {output_file}")


if __name__ == "__main__":
    run_generatetopic()