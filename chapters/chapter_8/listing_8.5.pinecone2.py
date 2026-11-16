from dotenv import load_dotenv
import os
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# 프로젝트 루트의 .env 를 환경변수로 로드
load_dotenv()

# get openai api key from platform.openai.com
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# find API key in console at app.pinecone.io
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# PINECONE_CONTROLLER_HOST 가 있으면 로컬 Pinecone(Docker 에뮬레이터)를 사용
PINECONE_HOST = os.getenv("PINECONE_CONTROLLER_HOST")
if PINECONE_HOST:
    pc = Pinecone(api_key=PINECONE_API_KEY, host=PINECONE_HOST)
else:
    pc = Pinecone(api_key=PINECONE_API_KEY)


# Set up vectorstore
index_name = "ch8-pincecone-llm-example"
# 로컬 임베딩 모델 (OpenAI text-embedding-ada-002 대체, 384차원)
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
text_field = "text"

# 인덱스가 없으면 생성하고 예제 문서를 채워 넣음
try:
    pc.describe_index(index_name)
except Exception:  # 인덱스가 없으면 새로 생성
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index_host = pc.describe_index(index_name).host
if PINECONE_HOST:
    index_host = "http://" + index_host
index = pc.Index(host=index_host)
if index.describe_index_stats().total_vector_count == 0:
    docs = [
        "Johannes Gutenberg (c. 1400 - 1468) was a German goldsmith and "
        "inventor who introduced movable-type printing to Europe around 1440.",
        "The Gutenberg Bible, printed in Mainz around 1455, was the first "
        "major book produced with movable metal type in Europe.",
        "The printing press spread rapidly across Europe and helped drive "
        "the Renaissance, the Reformation and the Scientific Revolution.",
    ]
    index.upsert(
        vectors=[
            {
                "id": f"doc{i}",
                "values": [float(x) for x in embedder.embed_query(d)],
                "metadata": {"text": d, "source": f"wikipedia-{i}"},
            }
            for i, d in enumerate(docs)
        ]
    )
    import time

    time.sleep(3)  # 로컬 인덱스가 반영될 때까지 잠시 대기
vectorstore = PineconeVectorStore(
    index=index, embedding=embedder, text_key=text_field
)

# Make a query
query = "Who was Johannes Gutenberg?"
print(
    vectorstore.similarity_search(
        query, k=3  # our search query  # return 3 most relevant docs
    )
)

# Now lets use these results to enrich our LLM prompt
# Set up LLM
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name=os.getenv("OLLAMA_MODEL", "gpt-3.5-turbo"),
    temperature=0.0,
    max_tokens=3000,
)

# Run query with vectorstore
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)
print(qa.run(query))

# Include wikipedia sources
qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)
result = qa_with_sources(query)
print(result)

# {'question': 'Who was Johannes Gutenberg?', 'answer': ' The image shows a list of births and deaths for the month of April in the year 1771. The births include Richard Trevithick, Karl Philipp Fürst zu Schwarzenberg, Jean Rapp, Prince Ernest Augustus of Great Britain, Sir Walter Scott, Archduke Charles of Austria, Emperor Kokaku of Japan, and Frederick William, Duke of Brunswick. The deaths include Jean-Andoche Junot and Marie François Xavier Bichat. ', 'sources': ''}
