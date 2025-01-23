import streamlit as st
import fitz  # PyMuPDF
import ollama
import chromadb

st.title("💬 Chat with Your PDF Documents using Llama 3.1 8B")

# Initialize ChromaDB Client
client = chromadb.Client()

# Check if the collection exists, otherwise create a new one
collection = None
try:
    collection = client.get_collection(name="docs")
    # st.write("Collection 'docs' exists. New data will be appended.")
except Exception as e:
    # st.write("No existing collection found. Creating a new one.")
    collection = client.create_collection(name="docs")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    # Convert the uploaded PDF file to a byte stream for fitz
    pdf_document = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text("text")
    return text

# File uploader for user to provide PDF document(s)
uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

# Track document upload status
if uploaded_file:
    st.write("Processing the PDF file...")
    
    # Extract text from the PDF file
    document_text = extract_text_from_pdf(uploaded_file)
    
    # Split the extracted text into smaller chunks (e.g., by paragraphs)
    documents = document_text.split("\n\n")  # Split by paragraph or any other delimiter
    
    # Generate embeddings for each document chunk and store them in ChromaDB collection
    for i, d in enumerate(documents):
        if d.strip():  # Only process non-empty paragraphs
            response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
            embedding = response["embedding"]
            # Add new documents and embeddings to the existing collection
            collection.add(
                ids=[str(i)],  # Ensure each document has a unique ID
                embeddings=[embedding],
                documents=[d]
            )
    st.success("PDF content successfully stored!")
    
    # Set the session state to indicate that the document has been uploaded
    st.session_state["pdf_uploaded"] = True
else:
    st.session_state["pdf_uploaded"] = False

# Initialize message history if not already present
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display the previous messages from the session
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
    else:
        st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

# Function to generate responses from the model using the retrieved document data
def generate_response(prompt):
    # Generate embedding for the user's question (prompt)
    response = ollama.embeddings(prompt=prompt, model="mxbai-embed-large")
    
    # Query the ChromaDB collection to find the most relevant document
    results = collection.query(query_embeddings=[response["embedding"]], n_results=1)
    
    # Retrieve the document
    if results['documents']:
        relevant_data = results['documents'][0][0]  # Get the most relevant document
        # Use the Llama model to generate a response combining the document and the prompt
        response = ollama.generate(
            model="llama3.1:8b",
            prompt=f"Using this data: {relevant_data}. Respond to this prompt: {prompt}"
        )
        return response['response']
    else:
        return "No relevant document found for the query."

# Display an initial message if the document has been uploaded
if st.session_state.get("pdf_uploaded"):
    if user_input := st.chat_input("Ask a question based on the uploaded PDF document:"):
        # Add the user message to the session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Display the user's message immediately in the chat
        st.chat_message("user", avatar="🧑‍💻").write(user_input)
        
        # Generate the assistant's response
        with st.chat_message("assistant", avatar="🤖"):
            response_message = generate_response(user_input)
            st.write(response_message)
        
        # Add the assistant's response to the session state
        st.session_state.messages.append({"role": "assistant", "content": response_message})
else:
    st.write("Please upload a PDF document to start chatting.")
