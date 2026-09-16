"""Adaptive evaluation v1. Freeze prompts/settings before policy tuning.

The last sixteen new cases are held out. Categories are report metadata only.
"""

SYSTEM = "You are a concise assistant."


def case(name, prompt, category="text", *, regex=None, schema=None, budget=512,
         system=SYSTEM, split="development", original=False):
    return dict(name=name, messages=[{"role": "system", "content": system},
                                    {"role": "user", "content": prompt}],
                category=category, regex=regex, json_schema=schema, max_tokens=budget,
                split=split, original=original)


CORPUS = [
    case("cuda_ready", "Reply with CUDA ready.", budget=256, original=True),
    case("gpu_summary", "In one concise sentence, explain what a GPU does.", budget=256, original=True),
    case("number_sequence", "Write the numbers one through ten, separated by commas.", budget=256, original=True),
    case("regex_cuda_ready", "Reply with CUDA ready.", regex="CUDA Ready", budget=256, original=True),
    case("json_required_enum", "Use no spaces or newlines in the JSON response.",
         schema={"type": "object", "properties": {"content": {"enum": ["CUDA ready", "Ready"]}},
                 "required": ["content"]}, system="Return compact JSON only.", budget=256, original=True),
    case("cache_explanation", "In two sentences, explain why a cache can make a program faster.", budget=256, original=True),
    case("python_function", "Write only a Python function that returns the largest even integer in a list, or None if there is none.", budget=256, original=True),
    case("extract_cities", "Extract the city names from this sentence, as a comma-separated list only: Mia visited Oslo in May, Kyoto in June, and Lima in July.", budget=256, original=True),
    case("support_reply", "Write a polite two-sentence reply to a customer whose parcel arrived two days late. Apologize and offer to refund shipping.", budget=256, original=True),
]

NEW_CASES = [
    ("yes_no", "Is seven an odd number? Answer yes or no only.", "short"),
    ("capital", "What is the capital of Japan? Reply with just the city.", "short"),
    ("arithmetic", "What is 17 plus 26? Reply with the number only.", "short"),
    ("opposite", "Give the antonym of hot. One word only.", "short"),
    ("translation", "Translate good morning into Spanish. Return only the translation.", "short"),
    ("classification", "Classify this review as positive or negative: The meal was wonderful. One word only.", "short"),
    ("abbreviation", "Expand CPU. Return only the expansion.", "short"),
    ("acknowledge", "Acknowledge receipt with exactly: Received.", "short"),
    ("cache_long", "Explain how a CPU cache works in three short paragraphs, including locality and a cache miss. Keep it under 180 words.", "prose"),
    ("rain", "Explain the water cycle to a ten-year-old in about 120 words.", "prose"),
    ("queue", "Explain when a queue is more useful than a stack. Give two examples in under 150 words.", "prose"),
    ("apology", "Write a 120-word apology for a cancelled community workshop, including rescheduling and refunds.", "prose"),
    ("story", "Write a 150-word story about a lighthouse keeper who finds a clock that runs backward.", "prose"),
    ("binary_search", "Write only an iterative Python binary search function returning an index or -1.", "code"),
    ("sql", "Write only SQL that groups orders(customer_id, amount) by customer_id and selects customers with total amount above 100.", "code"),
    ("deduplicate", "Write only a Python function to remove duplicates from a list while preserving order, plus three assert examples.", "code"),
    ("typescript", "Write only a TypeScript function that groups strings by their length using a Map.", "code"),
    ("emails", "Extract email addresses, one per line: Contact Ana at ana@example.com and Bo at bo@example.org. The meeting is Monday.", "extraction"),
    ("dates", "Extract dates in their original form, one per line: Built on 2020-03-12, inspected on 2021-07-09, and sold on 2024-11-02.", "extraction"),
    ("sort_names", "Sort these names alphabetically and output one per line: Zara, Omar, Alice, Chen, Beatrice, Diego.", "extraction"),
    ("csv", "Convert to CSV with columns name,age,city and no explanation: Ana is 31 and lives in Lima; Bo is 24 and lives in Oslo; Cy is 42 and lives in Perth.", "extraction"),
    ("count_then_explain", "First list integers 1 through 30 separated by commas. Then explain in 80 words why counting is useful in programming.", "changing"),
    ("explain_then_count", "Explain in 80 words why predictable patterns help compression. Then list integers 1 through 30 separated by commas.", "changing"),
    ("revision_advice", "In 120 words, explain how to revise a first draft of a short story.", "prose"),
    ("long_context", "Context: " + "The archive stores books on numbered shelves. Each shelf has a catalog. " * 160 + "\nSummarize this archive system in two sentences.", "prose"),
    ("comparison", "Compare bicycles and buses for a daily commute in about 130 words.", "prose"),
    ("merge_sorted", "Write only a Python function to merge two sorted lists without using sorted, followed by two assert examples.", "code"),
    ("javascript", "Write only a JavaScript function that counts word frequencies ignoring case, followed by one example call.", "code"),
    ("numbers_extract", "Extract only quantities with units: The crate weighs 12 kg, is 30 cm wide and costs 45 dollars. The truck travels 80 km.", "extraction"),
    ("long_extract", "Records: " + "Alice: inactive; Bob: inactive; " * 120 + "Nora: active; Wei: active. Return only the active names separated by commas.", "extraction"),
    ("copy_then_story", "First write abcdefghijklmnopqrstuvwxyz three times, each on its own line. Then write a 70-word story about a missing letter.", "changing"),
]
for i, (name, prompt, category) in enumerate(NEW_CASES):
    CORPUS.append(case(name, prompt, category, split="heldout" if i >= 23 else "development"))

for name, prompt, pattern in [
    ("year", "Return the year 2024.", "[0-9]{4}"),
    ("digits", "Return 32 digits, using the repeating pattern 1234567890.", "[0-9]{32}"),
    ("product_code", "Return product code ABC-1234.", "[A-Z]{3}-[0-9]{4}"),
    ("repeated", "Return the word ready followed by a space, sixteen times, then done.", "(ready ){16}done"),
]:
    CORPUS.append(case(name, prompt, "regex", regex=pattern, split="heldout"))

for name, prompt, schema in [
    ("boolean_json", "Return true as JSON.", {"type": "boolean"}),
    ("enum_json", "Choose the color blue and return it as JSON.", {"enum": ["red", "green", "blue"]}),
    ("object_json", "Return a compact JSON object with name Ada and age 36.",
     {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"], "additionalProperties": False}),
    ("array_json", "Return a JSON array containing the integers 1 through 12.",
     {"type": "array", "items": {"type": "integer"}, "minItems": 12, "maxItems": 12}),
]:
    CORPUS.append(case(name, prompt, "json", schema=schema, split="heldout"))

assert len(CORPUS) == 48
assert sum(c["split"] == "heldout" for c in CORPUS) == 16
