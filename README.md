WEEK 5 updates:

Original goals (i met all of them)
- get the data and retrieval pipeline working
- Compile an e-textbook (or facts sheet from online information if scraping doesn’t work)
- Create controlled vocab:
    - Classify animal conditions into their canonical name, and names that are often used/interchangeable
    - symptoms: do the same thing with symptoms (eg group ‘puking’ and ‘vomiting’)
- Chunk the text from textbook into 200-400 token passages with metadata(condition, species, section, page/url)
- Build a retriever that normalizes input symtoms via vocab and returns the top-k passages with citations (from the vet textbook).


Comments:
- i realized using an e-textbook was illegal so i instead scraped a list of dog diseases from the page "list of dog diseases" on wikipedia
and i opened each link to get their content and then it made it a little more complicated because i had to filter out irrelevant information
it wouldve been easier with a textbook bc i wouldnt have to do that and it would have consistent sections, but this challegne made it fun.

- i think it would be helpful if i could make my controlled vocab into a hierarchical graph to map relationships in the future (symptoms>disease) or something like that. I will aim to do that next week if  have time.
- instead of chunking into 200-400 token passages i did it by paragraph, but i also kept a paragraph.idx so that i could know which paragraphs are 'closer' to each otehr which might make them related.
- my document 'test.py' is currently working. if you run it and change the query to different prompts "dog is getting diarhea" or "dog is aggressive" it can return top k passages with citations from the wikipedia page.