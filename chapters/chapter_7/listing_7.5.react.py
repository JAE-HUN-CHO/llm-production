import os
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv()

# load API keys; you will need to obtain these if you haven't yet
# OPENAI_API_BASE / OPENAI_API_KEY 는 .env 의 로컬 Ollama 설정을 사용
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "ollama")
# SERPER_API_KEY 가 없으면 키가 필요 없는 DuckDuckGo 검색으로 대체
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# text-davinci-003 은 폐기되어 채팅 모델(로컬 Ollama)로 대체
llm = ChatOpenAI(
    model_name=os.getenv("OLLAMA_MODEL", "gpt-3.5-turbo"),
    temperature=0,
    max_tokens=int(os.getenv("MAX_TOKENS", "1500")),
    # 추론 모델(Ollama)의 생각 길이를 최소화
    model_kwargs={"extra_body": {"reasoning_effort": "none"}},
)
if SERPER_API_KEY:
    os.environ["SERPER_API_KEY"] = SERPER_API_KEY
    tools = load_tools(["google-serper", "llm-math"], llm=llm)
else:
    tools = load_tools(["llm-math"], llm=llm)
    tools.append(DuckDuckGoSearchRun())
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=int(os.getenv("MAX_ITERATIONS", "8")),
)

agent.run(
    "Who is Olivia Wilde's boyfriend? \
    What is his current age raised to the 0.23 power?"
)

# > Entering new AgentExecutor chain...
# I need to find out who Olivia Wilde's boyfriend is and then
# calculate his age raised to the 0.23 power.
# Action: Search
# Action Input: "Olivia Wilde boyfriend"
# Observation: Olivia Wilde started dating Harry Styles after ending
# her years-long engagement to Jason Sudeikis — see their relationship
# timeline.
# Thought: I need to find out Harry Styles' age.
# Action: Search
# Action Input: "Harry Styles age"
# Observation: 29 years
# Thought: I need to calculate 29 raised to the 0.23 power.
# Action: Calculator
# Action Input: 29^0.23
# Observation: Answer: 2.169459462491557

# Thought: I now know the final answer.
# Final Answer: Harry Styles, Olivia Wilde's boyfriend, is 29 years old
# and his age raised to the 0.23 power is 2.169459462491557.

# > Finished chain.

# "Harry Styles, Olivia Wilde's boyfriend, is 29 years old and his age
# raised to the 0.23 power is 2.169459462491557."
