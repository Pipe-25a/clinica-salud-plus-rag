import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

# 1. Cargar documentos de la clinica
loader = PyPDFLoader("data/clinica.pdf")
documentos = loader.load()

# 2. Dividir en chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documentos)

# 3. Crear embeddings y base vectorial
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. Crear el retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Definir el prompt del asistente
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Eres un asistente virtual de la Clinica Salud Plus.
Responde SOLO usando el contexto proporcionado.
Si no tienes informacion, di: "No tengo informacion disponible."
No inventes datos medicos bajo ninguna circunstancia.

Contexto: {context}
Pregunta: {question}
Respuesta:"""
)

# 6. Crear pipeline RAG
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template}
)

# 7. Hacer consultas de prueba
preguntas = [
    "Como prepararse para un examen de sangre?",
    "Cuales son los horarios de atencion?",
    "Que especialidades medicas tienen disponibles?"
]

for pregunta in preguntas:
    print(f"\nPregunta: {pregunta}")
    respuesta = chain.invoke({"query": pregunta})
    print(f"Respuesta: {respuesta['result']}")
