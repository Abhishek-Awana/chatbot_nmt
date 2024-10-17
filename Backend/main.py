
import os
from typing import Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SummaryIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core.query_engine import RouterQueryEngine


app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv('.env')

# Load translation model and tokenizer
translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
translation_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# API keys
groq_api_key = os.getenv("GROQ_API_KEY")
# Set up LLM and embedding model
llm = Groq(model="llama3-8b-8192")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
Settings.chunk_overlap = 20

# Pre-load documents and create storage context
documents = SimpleDirectoryReader(input_files=["Data/Digital_Transformation.pdf"]).load_data()
storage_context = StorageContext.from_defaults()
nodes = Settings.node_parser.get_nodes_from_documents(documents)
storage_context.docstore.add_documents(nodes)

# Create indices for summarization and vector search
summary_index = SummaryIndex(nodes, storage_context=storage_context)
vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
index = VectorStoreIndex.from_documents(documents)
# query_engine = index.as_query_engine(similarity_top_k=3)


# Define query engines for summarization and vector search
# summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize", use_async=True)
# vector_query_engine = vector_index.as_query_engine(similarity_top_k=3)

# Define tools for the query engines
# summary_tool = QueryEngineTool.from_defaults(
#     query_engine=summary_query_engine,
#     description="Useful for summarization of the content."
# )
# vector_tool = QueryEngineTool.from_defaults(
#     query_engine=vector_query_engine,
#     description="Useful for retrieving specific context from the documents."
# )

# Set up RouterQueryEngine to switch between different modes
# tools = [summary_tool, vector_tool]
# router_query_engine = Settings.router_query_engine_class(
#     tools=tools,
#     default_tool_name="vector_search",
# )

vector_tool = QueryEngineTool(
    index.as_query_engine(),
    metadata=ToolMetadata(
        name="vector_search",
        description="Useful for retrieving specific context from the documents."
    ),
)

summary_tool = QueryEngineTool(
    index.as_query_engine(response_mode="tree_summarize", use_async=True),
    metadata=ToolMetadata(
        name="summary",
        description="Useful for summarization of the content."
    ),
)

query_engine = RouterQueryEngine.from_defaults(
    [vector_tool, summary_tool], select_multi=False, verbose=True, llm=llm
)


def translate_input(text: str, target_lang_code: str) -> str:
    """Translates the given text or email to the specified target language."""
    translated_lines = []

    # Check if the input is multiline (email format)
    if "\n" in text:
        # Split the text into lines
        lines = text.strip().split("\n")
        
        for line in lines:
            # Preserve empty lines
            if line.strip() == "":
                translated_lines.append("")
            else:
                # Tokenize input line for translation
                tokenizer_input = translation_tokenizer(line, return_tensors="pt")
                
                # Generate translation specifying target language
                generated_tokens = translation_model.generate(
                    **tokenizer_input, 
                    forced_bos_token_id=translation_tokenizer.get_lang_id(target_lang_code)
                )
                
                # Decode and add the translated line to the list
                translated_line = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                translated_lines.append(translated_line)
        
        # Join the translated lines to form the final output
        translated_output = "\n".join(translated_lines)
    
    else:
        # If it's a single line of text, translate it directly
        tokenizer_input = translation_tokenizer(text, return_tensors="pt")
        
        # Generate translation specifying target language
        generated_tokens = translation_model.generate(
            **tokenizer_input, 
            forced_bos_token_id=translation_tokenizer.get_lang_id(target_lang_code)
        )
        
        # Decode the translated text
        translated_output = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    return translated_output

class QueryRequest(BaseModel):
    query: str
    language: str
    mode: str  # 'search', 'summarize', or 'translate'

@app.post("/query")
async def query_model(request: QueryRequest):
    try:
        query_text = request.query
        lang = request.language
        mode = request.mode
        # response = query_engine.query(query_text)
        # response = translate_input(query_text, lang) 
        # response = query_engine.query(query_text)

        # Select the appropriate query engine based on the mode
        if mode == "summarize":
            response = query_engine.query(query_text)
        elif mode == "search":
            response = query_engine.query(query_text)
        elif mode == "translate":
        #     # Directly translate without querying the vector or summary tools
            translated_response = translate_input(query_text, lang)
            return {"response": translated_response}
        else:
            raise HTTPException(status_code=400, detail="Invalid mode")

        # Convert the response to a string
        response_text = str(response)

        # Translate the response if needed
        if lang != "en":
            response_text = translate_input(response_text, lang)

        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying the model: {str(e)}")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

