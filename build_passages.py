from wikipedia_scrape import DOG_TERMS, mentions_dog_species
import wikipediaapi
import json, time
from pathlib import Path

UA = "vet-nlp-project/0.1 (contact: jxiong3@uchicago.edu)"
wiki = wikipediaapi.Wikipedia(language="en", user_agent=UA)

OUT = Path("data/passages.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

with open("data/kept_titles.json", "r", encoding="utf-8") as f:
    kept = json.load(f)
"""
I want this script to:
- Loop through each page in kept
- Divide it into passages
- Create a dictionary for each passage that looks like:
{"condition":"canine_parvovirus",
"species":"dog",
"section":"Clinical signs",
"paragraph_id": idx,
"source":"Wikipedia",
"license":"CC BY-SA 4.0",
"url":"https://en.wikipedia.org/?curid=12345"}
"""

# honestly it's probably more efficient to integrate this part with
# wikipedia_scrape since running it takes a long time, but you can
# do it later if you have time (can integrate species checks and stuff)

# helper functions
def title_is_canine(title: str) -> bool:
    """check if the title itself suggests a dog/canine disease"""
    t = title.lower()
    return ("canine" in t) or ("dog" in t)

def header_mentions_dog(text: str) -> bool:
    """check if header (first paragraph) mentions dog/canine"""
    header = (text or "").strip().lower().split("\n", 1)[0]
    return any(term in header for term in DOG_TERMS)

with OUT.open("w", encoding="utf-8") as f:
    for i, title in enumerate(kept, 1):
        try:
            page = wiki.page(title)
            if not page.exists():
                continue  # skip missing pages safely

            text = (page.text or "").strip()
            if not text:
                continue  # skip empty pages

            url = page.fullurl

            # check what species this is
            if title_is_canine(title) or header_mentions_dog(text):
                species = "dog"
            else:
                species = "mixed"

            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.split()) >= 20]  # list of paragraphs

            for idx, paragraph in enumerate(paragraphs):
                # skip records that don't mention dogs if the species is mixed
                if species == "mixed" and not mentions_dog_species(paragraph):
                    continue
                
                record = {
                    "condition": title,
                    "species": species,
                    "section": "unlabelled",  # i haven't implemented this yet but it could be useful for the future :D
                    "paragraph_id": idx,  # maybe i can use this later to get 'similar' paragraphs, like the ones before and after
                    "text": paragraph,
                    "source": "Wikipedia",
                    "license": "CC BY-SA 4.0",
                    "url": url
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # polite pause so you don't hit rate limits
            if i % 30 == 0:
                time.sleep(0.2)

        except Exception as e:
            print(f"[skip] {title}: {e}")  # don't let one bad page kill the run
            continue

print(f"Wrote passages to {OUT}")
