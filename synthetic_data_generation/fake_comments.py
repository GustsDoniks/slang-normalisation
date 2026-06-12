from openai import OpenAI
from dotenv import load_dotenv
import os
import csv
import json
import re
import time


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

SYSTEM_PROMPT = """You are an English internet slang data augmentation assistant.

Task:
Generate realistic but completely fictional English social media comments that naturally use a given slang term.

Input:
You will receive:
- target_term: the slang term that must appear naturally in the generated comments.
- target_meaning: the intended meaning of the slang term.
- target_example: an optional usage example from the slang dictionary.
- other_available_terms: other slang terms that may also be used if they fit naturally.

Generation rules:
- Generate exactly 5 comments.
- Each comment must include the target term or a very close inflected/cased version of it.
- Comments should sound like real short internet comments: casual, informal, sometimes messy, but still understandable.
- Comments may include other slang words from other_available_terms, but the target term must remain the main focus.
- Do not explain the slang term.
- Do not define the slang term.
- Do not write educational examples like a dictionary.
- Do not copy the provided example exactly.
- Do not include usernames, hashtags, URLs, or platform metadata.
- Comments should vary in tone and context.
- Comments can be short, like real comments: "Rare W", "this goes hard", "bro fell off".
- Mild profanity is allowed when natural, but avoid hate speech, slurs, sexual content, threats, or targeted harassment.
- Avoid political persuasion, real named public figures, real private people, and real current events.
- Make the comments fictional and generic.
- Keep each comment under 160 characters.

Return ONLY valid JSON in this exact shape:
{
  "comments": [
    "comment 1",
    "comment 2",
    "comment 3",
    "comment 4",
    "comment 5"
  ]
}
"""

def read_translations(filepath="translation.csv"):
    rows = []

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            term = row.get("term", "").strip()
            meaning = row.get("meaning", "").strip()
            example = row.get("example", "").strip()
            slang_type = row.get("type", "").strip()
            source = row.get("source", "").strip()

            if not term or not meaning:
                continue

            rows.append({
                "term": term,
                "meaning": meaning,
                "example": example,
                "type": slang_type,
                "source": source,
            })

    return rows

def generate_comments_for_term(target_row, all_terms, model="gpt-5.4-mini"):
    target_term = target_row["term"]
    target_meaning = target_row["meaning"]
    target_example = target_row.get("example", "")

    other_terms = [
        row["term"]
        for row in all_terms
        if row["term"].lower() != target_term.lower()
    ][:80]

    user_message = {
        "target_term": target_term,
        "target_meaning": target_meaning,
        "target_example": target_example,
        "other_available_terms": other_terms,
    }

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(user_message, ensure_ascii=False),
            },
        ],
        temperature=0.9,
    )

    raw = response.choices[0].message.content.strip()

    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        parsed = json.loads(raw)
        comments = parsed.get("comments", [])
    except json.JSONDecodeError:
        print(f"Could not parse JSON for term: {target_term}")
        print("Raw response:")
        print(raw)
        comments = []

    cleaned = []
    for comment in comments:
        if isinstance(comment, str):
            comment = comment.strip()
            if comment:
                cleaned.append(comment)

    return cleaned[:5]

def contains_target_term(comment, term):
    pattern = r"(?<![a-zA-Z0-9])" + re.escape(term.lower()) + r"(?![a-zA-Z0-9])"
    return re.search(pattern, comment.lower()) is not None

def main():
    input_file = "translation.csv"
    output_file = "synthetic_comments.csv"

    translations = read_translations(input_file)

    print(f"Loaded {len(translations)} terms from {input_file}")

    output_rows = []

    for i, row in enumerate(translations, 1):
        term = row["term"]
        meaning = row["meaning"]

        print(f"[{i}/{len(translations)}] Generating comments for: {term!r}")

        comments = generate_comments_for_term(row, translations)

        if len(comments) < 5:
            print(f"  Warning: only got {len(comments)} comments for {term!r}")

        for j, comment in enumerate(comments, 1):
            output_rows.append({
                "target_term": term,
                "target_meaning": meaning,
                "synthetic_comment": comment,
                "comment_number": j,
                "contains_target_term": contains_target_term(comment, term),
                "type": row.get("type", ""),
                "source": row.get("source", ""),
            })
        time.sleep(0.2)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "target_term",
            "target_meaning",
            "synthetic_comment",
            "comment_number",
            "contains_target_term",
            "type",
            "source",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nDone! Saved {len(output_rows)} synthetic comments to {output_file}")


if __name__ == "__main__":
    main()