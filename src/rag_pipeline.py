import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Configuracion GitHub Models
token = os.environ["GITHUB_TOKEN"]
base_url = os.environ["GITHUB_BASE_URL"]

# 1. Cargar documentos
loader = TextLoader("data/info_clinica.txt", encoding="utf-8")
documentos = loader.load()

# 2. Dividir en chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documentos)

# 3. Embeddings y base vectorial
embeddings = OpenAIEmbeddings(
    api_key=token,
    base_url=base_url,
    model="text-embedding-3-small"
)
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Eres un asistente virtual de la Clinica Salud Plus.
Responde SOLO usando el contexto proporcionado.
Si no tienes informacion di: No tengo informacion disponible.
No inventes datos medicos.

Contexto: {context}
Pregunta: {question}
Respuesta:"""
)

# 5. LLM
llm = ChatOpenAI(
    api_key=token,
    base_url=base_url,
    model="gpt-4o-mini",
    temperature=0
)

# 6. Pipeline RAG moderno
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Preguntas de prueba
preguntas = [
    "Como prepararse para un examen de sangre?",
    "Cuales son los horarios de atencion?",
    "Que especialidades medicas tienen disponibles?"
]

for pregunta in preguntas:
    print(f"\nPregunta: {pregunta}")
    respuesta = chain.invoke(pregunta)
    print(f"Respuesta: {respuesta}")
