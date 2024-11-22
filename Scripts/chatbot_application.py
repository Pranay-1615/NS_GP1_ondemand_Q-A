import gradio as gr
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from gpt4all import GPT4All
import requests


# Load the embedding model
embedder = SentenceTransformer("all-MiniLM-L12-v2")

# Connect to Qdrant
qdrant_client = QdrantClient(host="localhost", port=6333)

# Load the GPT4All model
model_path = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"  # Update this path if needed
gpt4all_model = GPT4All(model_path)

collection_name = "network_security_knowledge"

SERPAPI_API_KEY = "7fa7214ef761e09c624ba83150a6881f0238327672d3fe4023bc65610c566cc8"

relevance_threshold = 0.4 #choose relevance threshold to improve the output

def find_relevant_document(prompt):
    question_embedding = embedder.encode([prompt])[0]

    # Perform a search in the Qdrant database
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=question_embedding.tolist(), 
        )

    # Prepare the output
    relevant_pages = []
    for hit in search_results:
        if(hit.score>=relevance_threshold):
            payload = hit.payload
            relevant_pages.append({
                "document_name": payload["document"],
                "page_number": payload["page_number"],
                "reference": payload["text"]
                })

    return relevant_pages

def web_search(query):
    """Perform a web search using SerpAPI."""
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "engine": "google",
        "num" : 1
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        search_results = response.json().get('organic_results', [])
        # Extract the title and snippet from the search results
        message = ""
        for result in search_results:
            message +=f"{result['title']}-[URL:{result['link']}]\n"
            data = f"{result['snippet']}\n"
        return [data,message]
    else:
        print("Error with web search API:", response.status_code)
        return ["Internet Search Failure.","Error"]


def generate_response(prompt):
    
    # Find the relevant-documents
    relevant_pages = find_relevant_document(prompt)
    
    if relevant_pages:
        context_prompt = f"Answer the following question '{prompt}'\n\n from the below Context:\n"
        for i in relevant_pages:
            context_prompt += f"{i['reference']}\n\n"
        response = gpt4all_model.generate(context_prompt)
        source = "\n".join([f"Document: {page['document_name']}, Page: {page['page_number']}\nReference: {page['reference']}\n" for page in relevant_pages])
    else:
        source = "No relevant information found in the documents, Searching from Internet\n"
        resp1 = web_search(prompt)
        response = resp1[0]
        source+=resp1[1]

    return response, source
    
    

iface = gr.Interface(
    fn= generate_response,
    inputs = gr.Textbox(label="Enter your Query?",lines=5),
    outputs = [gr.Textbox(label="Chatbot response",lines=20),gr.Textbox(label="Sources",lines=10)],
    title = "On-demand Professor Q&A Chatbot"
    )

iface.launch()
