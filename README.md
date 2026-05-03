# Clínica Salud Plus — Asistente RAG
Sistema de atención al paciente basado en LLMs y RAG.
Responde preguntas frecuentes usando documentos internos de la clínica.
## Integrantes
- Felipe Pérez S.
- Ignacio Naum F.
## ¿Qué hace este sistema?
1. El paciente hace una pregunta
2. El sistema busca información en los documentos de la clínica
3. El LLM genera una respuesta basada en esa información
4. Se evitan alucinaciones al responder solo con datos reales
## Requisitos
- Python 3.10 o superior
- Una API Key de OpenAI
## Instalación
1. Clona el repositorio:
   git clone https://github.com/tu-usuario/clinica-salud-plus-rag
2. Instala las dependencias:
   pip install -r requirements.txt
3. Crea un archivo .env con tu API Key:
   OPENAI_API_KEY=tu_clave_aqui
## Ejecución
   python src/rag_pipeline.py
## Estructura 
- src/rag_pipeline.py → Código principal del RAG
- data/info_clinica.txt → Documentos de la clínica
- requirements.txt → Dependencias necesarias
## Tecnologías usadas
- LangChain → Framework para el RAG
- ChromaDB → Base de datos vectorial
- OpenAI GPT-4o-mini → Modelo 
- Python → Lenguaje de programación
