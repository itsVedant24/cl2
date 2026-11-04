import re

# Sample documents
documents = {
    "D1": "I love machine learning",
    "D2": "Machine learning is interesting",
    "D3": "I love Python programming"
}

# Create inverted index
inverted_index = {}

for doc_id, text in documents.items():

    # Convert text to lowercase
    words = re.findall(r'\b\w+\b', text.lower())

    for word in words:

        if word not in inverted_index:
            inverted_index[word] = []

        if doc_id not in inverted_index[word]:
            inverted_index[word].append(doc_id)

# Display inverted index
print("Inverted Index:")

for word, doc_list in inverted_index.items():
    print(word, ":", doc_list)

# Search document
query = input("\nEnter search word: ").lower()

print("\nRetrieved Documents:")

if query in inverted_index:
    for doc_id in inverted_index[query]:
        print(doc_id, ":", documents[doc_id])
else:
    print("No documents found!")

# 2. Document Retrieval Using Inverted Files
# An inverted file or inverted index is a data structure used to retrieve documents quickly. It stores each word along with the document IDs in which that word appears. When a user searches for a particular word, the system checks the inverted index and displays the matching documents. It is commonly used in search engines and information 
# retrieval systems.or inverted index is a data structure used to retrieve documents quickly. It stores each word along with the document IDs in which that word appears. When a user searches 
# for a particular word, the system checks the inverted index and displays the matching documents. It is commonly used in search engines and information retrieval systems.