import requests
import wikipediaapi
import json

UA = "vet-nlp-project/0.1 (contact: jxiong3@uchicago.edu)"
API = "https://en.wikipedia.org/w/api.php"

wiki = wikipediaapi.Wikipedia(language="en", user_agent=UA)

def get_wiki_links(title: str):
    session = requests.Session()
    params = {
        "action": "query",
        "format": "json",
        "prop": "links",
        "titles": title,        # page title
        "pllimit": "max",       # give as many links as possible
        "plnamespace": "0",     # only get the actual articles
        "redirects": "1",       # follow redirects (aka shortcut pages)
        "formatversion": "2"
    }
    links = []

    while True:
        r = session.get(API, params=params, headers={"User-Agent": UA}, timeout=20)
        r.raise_for_status()  # raise if not 200
        try:
            data = r.json()
        except ValueError:
            # Debug aid: print first part of response to see HTML error page
            print("Non-JSON response head:\n", r.text[:500])
            raise

        pages = data.get("query", {}).get("pages", [])
        if pages and "links" in pages[0]:
            links.extend(link["title"] for link in pages[0]["links"])

        # handle pagination
        cont = data.get("continue")
        if not cont:
            break
        # plcontinue is used for the next page of links
        params["plcontinue"] = cont.get("plcontinue")

    return links

#Use the exact page title
page_title = "List of dog diseases"
print("Page title check:")
print(wiki.page(page_title).title)  # should print "List of dog diseases"

dog_disease_links = get_wiki_links(page_title)
print(f"Found {len(dog_disease_links)} links.")
print(dog_disease_links[:20]) #print like the first 20

# Retrieved list of links in page "list of dog diseases"
# Problem: some of these are not actual disease articles.
# Also, some of these disease links are to generic human diseases
# Solution: create a key-word filter. 

#first, get all the content from the links

DISEASE_TERMS = [
    "disease","disorder","infection","syndrome","condition",
    "cancer","tumor","neoplasm","encephalitis","arthritis",
    "dermatitis","myositis","enteritis","nephritis","panleukopenia",
    "parvovirus","distemper","rabies","influenza","giardiasis",
    "toxoplasmosis","leptospirosis","bloat","gdv","heartworm", "canine",
    "virus"
]
DOG_TERMS = ["dog", "canine", "puppy", "puppies", "dogs"]

def looks_like_disease_title(title: str) -> bool:
    "Function that goes through DISEASE_TERMS returns if they exist in the article "
    t = title.lower()
    return any(k in t for k in DISEASE_TERMS)

def looks_like_disease_lead(text: str) -> bool:
    """Keep if first paragraph says 'is a ... disease/disorder/infection/...'"""
    lead = (text or "").strip().lower().split("\n", 1)[0]
    triggers = [" is a ", " are ", " is an "]
    disease_words = ["disease", "disorder", "infection", "syndrome", "condition"]
    return any(trig in lead for trig in triggers) and any(dw in lead for dw in disease_words)

def mentions_dog_species(text: str) -> bool:
    "return True if article mentions dog"
    if not text:
        return False
    full_text= (text or "").strip().lower()
    return any(term in full_text for term in DOG_TERMS)

kept = []
skipped = []

if __name__ == "__main__":
    for title in dog_disease_links:
        page = wiki.page(title)
        if looks_like_disease_title(title):
            if page.exists() and mentions_dog_species(page.text):
                kept.append(title)
                continue

        # If title doesn't look disease-y, fall back to a quick lead check:
        if page.exists() and looks_like_disease_lead(page.text) and mentions_dog_species(page.text):
            kept.append(title)
        else:
            skipped.append(title)

        print(f"Kept {len(kept)} probable diseases; skipped {len(skipped)} others.")

    print("Sample kept:", kept[:10]) #intuition check
    print("Sample skipped:", skipped[:10])

    with open("data/kept_titles.json", "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)