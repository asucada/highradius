# Importing Dependencies
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Dataset Directory Path
DATASET = "dataset/"

# Faiss Index Path
CHROMA_INDEX = "vectorstore/"

# Create Vector Store and Index
def embed_all():

    # Create the document loader
    loader = CSVLoader(file_path="./dataset/ghl-support-docs.csv")
    # Load the documents
    documents = loader.load()
    # Create the splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    # Split the documents into chunks
    chunks = splitter.split_documents(documents)
    # Load the embeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
    )
    # Create the vector store
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_INDEX)

if __name__ == "__main__":
    embed_all()