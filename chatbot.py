import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

CHROMA_PATH = "C:/Users/Ritam Choudhury/rag-chatbot/chroma_db"

PROMPT_TEMPLATE = """
Answer the question based ONLY on the following context:

{context}

---

Question: {question}

Answer:
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if len(results) == 0 or results[0][1] < 0.3:
        print("Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.invoke({"context": context_text, "question": query_text})
    
    model = ChatOllama(model="phi3", temperature=0.7)
    response_message = model.invoke(prompt)
    response_text = response_message.content
    
    print(response_text)

if __name__ == "__main__":
    main()
