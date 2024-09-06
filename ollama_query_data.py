import argparse
from langchain_chroma import Chroma # yes
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_ollama_embedding import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---
Do not use your internal knowledge and do not fabricate information.
If you don't know the answer, state that you are unable to answer the 
question with the provided context.
Answer the question based on the above context: {question}
"""


def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)

def query_rag(query_text: str):
    # Initialize the Chroma database with OpenAI Embedding function.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the Chroma database for the Top 5 most similiar matches.
    results = db.similarity_search_with_score(query_text, k=5)

    # Format the context text and plug into prompt template.
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # Initialize the LLM.
    model = Ollama(model="llama3")

    # Receive response from LLM.
    response_text = model.invoke(prompt)

    # Format the output with the response content and sources of context
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

    return response_text

if __name__ == "__main__":
    main()