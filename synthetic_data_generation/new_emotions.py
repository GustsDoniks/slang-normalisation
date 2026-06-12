from openai import OpenAI
from dotenv import load_dotenv
import os
import csv
import json
import re
import time
import random
from typing import List, Dict, Any

CONFIG = {
    "INPUT_TRANSLATIONS_CSV": "translation.csv",
    "OUTPUT_CSV": "synthetic_normalized_comments_balanced.csv",

    "MODEL": "gpt-5.4-mini",

    "EMOTIONS": ["joy", "disgust", "surprise", "anger", "sadness", "fear"],
    "TARGET_PER_EMOTION": 50,

    "EXAMPLES_PER_API_CALL": 6,

    "LOWERCASE_OUTPUT": True,

    "REQUIRE_TARGET_TERM": True,

    "MAX_RETRIES_PER_TERM": 2,
    "MAX_TOTAL_API_CALLS": 1000,

    "SLEEP_SECONDS": 0.2,

    "SEED": 42,
}


random.seed(CONFIG["SEED"])

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are an English internet slang normalization dataset generator.

Task:
Generate realistic but fictional English social media comments that use a given slang term,
then provide the normalized standard-English version and an emotion label.

Input:
You will receive:
- target_term: the slang term that should appear in the original comment.
- target_meaning: the intended meaning of the slang term.
- requested_emotions: the exact emotion labels needed for this batch.
- other_available_terms: optional extra slang terms that may be used naturally.

You must generate exactly one example for each requested emotion.

Each example must contain:
1. original_comment:
   - A slang-heavy, realistic, fictional social media comment.
   - The target term must appear naturally, or a close inflected/cased/stretched version may appear.
   - It should sound like a real short internet comment, not a dictionary example.
   - It may contain emojis if natural.
   - It may contain other slang if natural.
   - Keep under 160 characters.

2. normalized_comment:
   - A clear standard-English normalization of the original comment.
   - Preserve the meaning, tone, and emotion.
   - Replace slang with standard English.
   - Do not explain the slang.
   - Do not say "this means".
   - Do not define words.
   - Do not add unrelated context.
   - Keep it as a comment, not a formal explanation.

3. emotion:
   - Must be exactly one of the requested emotions.
   - The emotion must match the emotional tone of the original comment.

Emotion meanings:
- joy: happiness, approval, excitement, amusement, admiration
- disgust: dislike, cringe, disapproval, embarrassment, contempt
- surprise: shock, disbelief, amazement, unexpected reaction
- anger: frustration, annoyance, outrage, irritation
- sadness: disappointment, hurt, regret, loneliness
- fear: anxiety, worry, nervousness, panic, concern

Style rules:
- Make comments fictional and generic.
- Do not include usernames, hashtags, URLs, platform metadata, or real private people.
- Avoid real named public figures and real current events.
- Mild profanity is allowed only if natural.
- Do not include hate speech, slurs, explicit sexual content, threats, or targeted harassment.
- Do not copy any provided wording exactly.
- Vary tone and context across examples.

Important:
- The normalized_comment should NOT be a dictionary definition.
- The normalized_comment should be what a normal person would write after replacing slang.
- The original_comment should be slang-heavy.
- The normalized_comment should be clearer and more standard-English than original_comment.

Return ONLY valid JSON in this exact shape:
{
  "examples": [
    {
      "original_comment": "slang-heavy comment",
      "normalized_comment": "standard-English normalized comment",
      "emotion": "joy"
    }
  ]
}
"""

def read_translations(filepath: str) -> List[Dict[str, str]]:
    rows = []

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            term = row.get("term", "").strip()
            meaning = row.get("meaning", "").strip()

            if not term or not meaning:
                continue

            rows.append({
                "term": term,
                "meaning": meaning,
                "example": row.get("example", "").strip(),
                "type": row.get("type", "").strip(),
                "source": row.get("source", "").strip(),
            })

    return rows


def clean_json_response(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def maybe_lowercase(text: str) -> str:
    text = normalize_space(text)

    if CONFIG["LOWERCASE_OUTPUT"]:
        return text.lower()

    return text


def contains_target_term(comment: str, term: str) -> bool:

    comment_lower = str(comment).lower()
    term_lower = str(term).lower().strip()

    if not term_lower:
        return False

    if term_lower in comment_lower:
        return True

    pattern = r"(?<![a-zA-Z0-9])" + re.escape(term_lower) + r"(?![a-zA-Z0-9])"
    return re.search(pattern, comment_lower) is not None


def is_valid_emotion(emotion: str) -> bool:
    return str(emotion).strip().lower() in CONFIG["EMOTIONS"]


def is_definition_like(text: str) -> bool:

    text = str(text).lower()

    bad_patterns = [
        r"\bmeans\b",
        r"\bmeaning\b",
        r"\bdefinition\b",
        r"\bstands for\b",
        r"\bis used to\b",
        r"\bused to describe\b",
        r"\brefers to\b",
    ]

    return any(re.search(pattern, text) for pattern in bad_patterns)


def validate_example(example: Dict[str, Any], target_term: str, needed_emotions: set) -> bool:
    if not isinstance(example, dict):
        return False

    original = normalize_space(example.get("original_comment", ""))
    normalized = normalize_space(example.get("normalized_comment", ""))
    emotion = str(example.get("emotion", "")).strip().lower()

    if not original or not normalized or not emotion:
        return False

    if len(original) > 200:
        return False

    if emotion not in needed_emotions:
        return False

    if not is_valid_emotion(emotion):
        return False

    if CONFIG["REQUIRE_TARGET_TERM"] and not contains_target_term(original, target_term):
        return False

    if is_definition_like(normalized):
        return False

    return True


def clean_example(example: Dict[str, Any]) -> Dict[str, str]:
    return {
        "original_comment": maybe_lowercase(example["original_comment"]),
        "normalized_comment": maybe_lowercase(example["normalized_comment"]),
        "emotion": str(example["emotion"]).strip().lower(),
    }


def get_counts(rows: List[Dict[str, str]]) -> Dict[str, int]:
    counts = {emotion: 0 for emotion in CONFIG["EMOTIONS"]}

    for row in rows:
        emotion = row["emotion"]
        if emotion in counts:
            counts[emotion] += 1

    return counts


def get_needed_emotions(rows: List[Dict[str, str]]) -> List[str]:
    counts = get_counts(rows)

    needed = []

    for emotion in CONFIG["EMOTIONS"]:
        remaining = CONFIG["TARGET_PER_EMOTION"] - counts[emotion]

        if remaining > 0:
            needed.extend([emotion] * remaining)

    return needed


def print_progress(rows: List[Dict[str, str]]) -> None:
    counts = get_counts(rows)

    total_target = CONFIG["TARGET_PER_EMOTION"] * len(CONFIG["EMOTIONS"])
    total_current = len(rows)

    print()
    print(f"Progress: {total_current}/{total_target}")

    for emotion in CONFIG["EMOTIONS"]:
        print(f"  {emotion}: {counts[emotion]}/{CONFIG['TARGET_PER_EMOTION']}")

    print()


def save_rows(rows, output_file):

    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["original_comment", "normalized_comment", "emotion"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)

def generate_examples_for_term(
    target_row: Dict[str, str],
    all_terms: List[Dict[str, str]],
    requested_emotions: List[str],
) -> List[Dict[str, str]]:

    target_term = target_row["term"]
    target_meaning = target_row["meaning"]

    other_terms = [
        row["term"]
        for row in all_terms
        if row["term"].lower() != target_term.lower()
    ]

    random.shuffle(other_terms)
    other_terms = other_terms[:80]

    user_message = {
        "target_term": target_term,
        "target_meaning": target_meaning,
        "requested_emotions": requested_emotions,
        "other_available_terms": other_terms,
    }

    needed_emotions_set = set(requested_emotions)

    response = client.chat.completions.create(
        model=CONFIG["MODEL"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(user_message, ensure_ascii=False),
            },
        ],
        temperature=0.9,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    raw = clean_json_response(raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  JSON parse failed for term {target_term!r}")
        print(raw[:500])
        return []

    raw_examples = parsed.get("examples", [])

    if not isinstance(raw_examples, list):
        print(f"  Invalid JSON shape for term {target_term!r}")
        return []

    cleaned = []

    for example in raw_examples:
        if validate_example(example, target_term, needed_emotions_set):
            cleaned.append(clean_example(example))

    return cleaned

def main():
    input_file = CONFIG["INPUT_TRANSLATIONS_CSV"]
    output_file = CONFIG["OUTPUT_CSV"]

    translations = read_translations(input_file)

    print(f"Loaded {len(translations)} terms from {input_file}")
    print(f"Target: {CONFIG['TARGET_PER_EMOTION']} examples per emotion")
    print(f"Output file: {output_file}")

    output_rows = []
    seen_originals = set()

    api_calls = 0
    term_index = 0

    while True:
        needed_emotions = get_needed_emotions(output_rows)

        if not needed_emotions:
            print()
            print("Balanced target reached.")
            break

        if api_calls >= CONFIG["MAX_TOTAL_API_CALLS"]:
            print()
            print("Stopped: MAX_TOTAL_API_CALLS reached.")
            break

        random.shuffle(needed_emotions)
        requested_emotions = needed_emotions[:CONFIG["EXAMPLES_PER_API_CALL"]]

        target_row = translations[term_index % len(translations)]
        term_index += 1

        target_term = target_row["term"]

        print(
            f"[call {api_calls + 1}] term={target_term!r}, "
            f"requested_emotions={requested_emotions}"
        )

        collected_this_call = []

        for attempt in range(1, CONFIG["MAX_RETRIES_PER_TERM"] + 1):
            examples = generate_examples_for_term(
                target_row=target_row,
                all_terms=translations,
                requested_emotions=requested_emotions,
            )

            for example in examples:
                emotion = example["emotion"]

                current_counts = get_counts(output_rows)

                if current_counts[emotion] >= CONFIG["TARGET_PER_EMOTION"]:
                    continue

                key = example["original_comment"].lower().strip()

                if key in seen_originals:
                    continue

                seen_originals.add(key)
                output_rows.append(example)
                collected_this_call.append(example)

            if collected_this_call:
                break

            print(f"  attempt {attempt}: no usable examples")

        api_calls += 1

        if api_calls % 10 == 0:
            print_progress(output_rows)
            save_rows(output_rows, output_file)
            print(f"Intermediate save: {len(output_rows)} rows")

        time.sleep(CONFIG["SLEEP_SECONDS"])

    final_rows = []
    final_counts = {emotion: 0 for emotion in CONFIG["EMOTIONS"]}

    for row in output_rows:
        emotion = row["emotion"]

        if final_counts[emotion] < CONFIG["TARGET_PER_EMOTION"]:
            final_rows.append(row)
            final_counts[emotion] += 1

    save_rows(final_rows, output_file)

    print_progress(final_rows)
    print(f"Done. Saved {len(final_rows)} rows to {output_file}")


if __name__ == "__main__":
    main()