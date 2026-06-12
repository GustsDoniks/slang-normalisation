from openai import OpenAI
from dotenv import load_dotenv
import os
import csv
import json
import re

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def build_slang_map(filepath="translation.csv"):
    slang_map = {}
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = row["term"].strip()
            meaning = row.get("meaning", "").strip()
            example = row.get("example", "").strip()
            slang_map[term.lower()] = {
                "term": term,
                "meaning": meaning,
                "example": example,
            }
    return slang_map

def find_slang_in_comment(comment, slang_map):
    found = []
    seen = set()
    for term_lower, entry in slang_map.items():
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(term_lower) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, comment.lower()):
            if term_lower not in seen:
                seen.add(term_lower)
                found.append(entry)
    return found

SYSTEM_PROMPT = """You are an English internet slang normalization assistant.

Task: rewrite an informal/slang-heavy English social media comment into clear standard English, then assign one primary emotion.

Input:
* `original_comment`: raw social media comment.
* `relevant_translations`: candidate slang, abbreviation, emoji, or phrase meanings.

Use `relevant_translations` as guidance, not as a strict replacement table. A matched term may have a literal meaning or another non-slang meaning. Apply a slang meaning only when the full comment context supports it.

Examples:
* "I saw a goat on the farm" → goat is the animal, not "greatest of all time."
* "this song is fire" → fire means "very good."
* "the house is on fire" → fire means literal burning.
* "he got his bread up" → bread means money/income.
* "I bought bread" → bread is literal food.

Normalization rules:
* Normalize the meaning of the full comment.
* Preserve intended meaning, tone, sarcasm, humor, praise, criticism, shock, anger, sadness, and emoji meaning where reasonably inferable.
* Expand slang, abbreviations, censored spellings, emojis, and meme phrases into understandable standard English.
* Use translation examples only as usage guidance. Do not copy examples unless the same wording appears in the original comment.
* If clear slang appears but is not in `relevant_translations`, still normalize it if the meaning is obvious from context, lower confidence.
* If the comment is unclear, too short, or context-dependent, use the most reasonable interpretation and lower confidence.

Emotion: Choose exactly one label: `Anger`, `Disgust`, `Fear`, `Joy`, `Sadness`, `Surprise`.

Guidance:
* `Joy`: amusement, laughter, approval, excitement, admiration, hype.
* `Anger`: irritation, hostility, rage, resentment, aggressive criticism.
* `Disgust`: cringe, contempt, rejection, insults, moral disapproval, strong dislike.
* `Fear`: anxiety, worry, threat, panic, dread.
* `Sadness`: grief, heartbreak, disappointment, regret, loneliness.
* `Surprise`: shock, disbelief, amazement, confusion, unexpectedness.

`confidence` is an integer from 0 to 100 showing confidence that the normalized comment preserves the intended meaning and that the emotion label fits.

Respond ONLY with a valid JSON object — no markdown, no explanation — in this exact shape:
{
  "normalized_comment": "...",
  "emotion": "...",
  "confidence": 0
}"""


def normalize_comment(original_comment, relevant_translations):
    translations_text = ""
    if relevant_translations:
        lines = []
        for entry in relevant_translations:
            line = f"- {entry['term']}: {entry['meaning']}"
            if entry["example"]:
                line += f" (example: {entry['example']})"
            lines.append(line)
        translations_text = "\n".join(lines)
    else:
        translations_text = "(none detected)"

    user_message = (
        f"original_comment: {original_comment}\n\n"
        f"relevant_translations:\n{translations_text}"
    )

    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "normalized_comment": raw,
            "emotion": "Unknown",
            "confidence": 0,
        }

    return result

def main():
    slang_map = build_slang_map("translation.csv")
    print(f"Loaded {len(slang_map)} slang terms.\n")

    output_rows = []

    with open("synthetic_comments_only.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        comments = [row["comment_text"] for row in reader]

    for i, comment in enumerate(comments, 1):
        print(f"[{i}/{len(comments)}] Processing: {comment!r}")

        relevant = find_slang_in_comment(comment, slang_map)
        if relevant:
            print(f"  Slang detected: {[e['term'] for e in relevant]}")

        result = normalize_comment(comment, relevant)

        output_rows.append({
            "original_comment": comment,
            "normalized_comment": result.get("normalized_comment", ""),
            "emotion": result.get("emotion", ""),
            "confidence": result.get("confidence", ""),
        })

    with open("output.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["original_comment", "normalized_comment", "emotion", "confidence"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print("\nDone! Saved to output.csv")


if __name__ == "__main__":
    main()