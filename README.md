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



WEEK 7 updates:

Original goals:
- Train a classifier trained on synthetic data I make (or structured data - I have found a few datasets that i’ll probably have to clean and compile into one big dataset).
- Start with a logistic regression/Naive Bayes (interpretable coefficients)

- Create a rule-based scorer (make sure that species match, and the symptoms overlap) and combine all 3 components (retrieval + rules + classifier) into a blended model. So for example it would boost if species match, or symptoms overlap with the textbook. In week 7 I will just manually put in fixed weights (like say they are all equally important)
- Compute a top 3 accuracy on the synthetic dataset, a precision@k for retrieval and a confusion matrix for the classifier.
- Output: create a short table (retrieval-only vs classifier only vs rules only vs blended with weights). it would be successful if the blended weights beat the retrieval only Top 3.

COMMENTS:
- I decided to use multinomial logistic regression because Bayes relies on the assumption that symptoms are independent of each other, which is NOT true

- I had to spend a lot of time tuning the logistic regression. At first, with only the symptoms I had, the accuracy was 43.8%. I believe this is because after a 75/25 train test split, some classes only had 1-2 test samples left which made it very unreliable. Therefore, I increased the minimum samples required from 2 to 5, which increased the accuracy to 46.7%. I then tried different weights of C (the regularization strength)
and found that increasing to 5 from 1 increased accuracy to 53.3%. 
- I also tried feature engineering to create more interaction terms, which would make sense because certain symptoms happening together would be more indicative of specific diseases. However, it did not improve accuracy at all, so i removed it after. 
- I also tried using XGBoost and Random forests which brought the accuracy to 60% but Chelsea specifically wanted something interpretable so I didn't use them in the end.


- an additional thing I had to do this week was to fix the input formatting: the dataset I trained on was classified on different variable types such as Gender, Age, Species, breed, etc. so I had to create a structured converter that takes in a user's query and spits out an input that the classifier would expect. 

- One of the goals for the rule based scorer was to make sure symptoms overlap. I did not do this because I used fixed equal weights (1/3 each component) as a baseline. Symptom overlap is already captured by the retrieval system (BM25 textbook matching) and classifier (learned symptom patterns), so adding it to rules would be redundant at this stage. I think it's a better idea if I add this during weight optimization which is my goal for next week because I can tune how much weight by experimenting with different combinations. 

- I felt a bit ahead so i also went ahead and started on the final checkpoint which is checking the weights, so in run_evaluation.py I also added a test for testing different weight combinations. It boosts the accuracy if the retrieval is weighted heavier. However, i will definitely keep looking into that I think I still need to add more data and possibly test some edge cases.

How to run: run the script for run_evaluation.py