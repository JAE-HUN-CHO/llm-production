import os
import json
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from datasets import Dataset
import tiktoken

# 로컬 서버(Ollama 등)를 쓰려면 .env 의 OPENAI_API_BASE / OPENAI_API_KEY 사용
os.environ.setdefault("OPENAI_API_KEY", "Your API Key")
os.environ.setdefault(
    "OPENAI_API_BASE", "http://0.0.0.0:1234/v1"
)  # Replace with your server's address and port

llm = ChatOpenAI(
    model_name=os.getenv("OLLAMA_MODEL", "gpt-3.5-turbo"),  # Can be anything
    temperature=0.25,
    openai_api_base=os.environ["OPENAI_API_BASE"],  # Again
    openai_api_key=os.environ["OPENAI_API_KEY"],
    max_tokens=4000,
    # 추론(reasoning) 모델이 content 를 비우지 않도록 (Ollama) 생각 노력 최소화
    model_kwargs={"extra_body": {"reasoning_effort": "none"}},
    n=1,
)

# Embeddings for RAG
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# Tokenization for checking context length quickly
tiktoker = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Change the prompt to be whatever you want
prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        ###Instruction:
        You are an expert python developer.
        Given a question, some conversation history, and the closest code snippet we could find for the request, give your best suggestion for how to write the code needed to answer the User's question.

        ###Input:

        #Question: {question}
        
        #Conversation History: {conversation_history}
        
        Code Snippet:
        {code_snippet}


        ###Response:
        """

# Here's a rudimentary vectorDB, change it to anything we've covered
DATA_CSV = os.getenv(
    "EMBEDDINGS_CSV",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "data",
        "Slack_Dataset_embedded.csv",
    ),
)
# CSV 에서는 embedding 이 문자열로 저장되므로 float 리스트로 변환
df = pd.read_csv(DATA_CSV)[["text", "embedding"]]


def _parse(v):
    try:
        e = json.loads(v)
        return e if len(e) == 384 else None
    except Exception:
        return None  # 깨진 행은 건너뜀


df["embedding"] = df["embedding"].apply(_parse)
df = df.dropna(subset=["embedding"]).reset_index(drop=True)
vectorDB = Dataset.from_pandas(df)
# If you haven't created a faiss or elasticsearch or usearch index, do it
try:
    vectorDB.load_faiss_index("embedding", "my_index.faiss")
except Exception:
    print("No faiss index, building one with add_faiss_index")
    vectorDB.add_faiss_index(column="embedding")

# To keep track of chat history, do this however you need!
message_history = []

# Search the Vector DB
query = "How can I train an LLM from scratch?"
embedded = embedder.encode(query)
q = np.array(embedded, dtype=np.float32)
_, retrieved_example = vectorDB.get_nearest_examples("embedding", q, k=1)

# Format the prompt
formatted_prompt = PromptTemplate(
    input_variables=["question", "conversation_history", "code_snippet"],
    template=prompt_template,
)
# Set up the actual LLM Chain
chain = LLMChain(llm=llm, prompt=formatted_prompt)

# Don't overload your context length
num_tokens = len(
    tiktoker.encode(
        f"{prompt_template},\n" + "\n".join(message_history) + query
    )
)
while num_tokens >= 4000:
    message_history.pop(0)
    num_tokens = len(
        tiktoker.encode(
            f"{prompt_template},\n" + "\n".join(message_history) + query
        )
    )

# Run RAG with your API
res = chain.run(
    {
        "question": query,
        "conversation_history": message_history,
        "code_snippet": retrieved_example["text"][0],
    }
)
message_history.append(f"User: {query}\nLlama: {res}")

# I'm just printing, do whatever you need to here
print(res)
