import os

from dotenv import load_dotenv
from openai import OpenAI

# 프로젝트 루트의 .env 를 찾아 환경변수로 로드 (상위 디렉터리까지 탐색)
load_dotenv()

# Load your API key from an environment variable
client = OpenAI(
    # This isn't technically needed as we are passing in the default key
    api_key=os.getenv("OPENAI_API_KEY"),
    # OPENAI_BASE_URL 환경변수(예: 로컬 Ollama)가 있으면 자동 사용됨
    base_url=os.getenv("OPENAI_BASE_URL"),
)

chat_completion = client.chat.completions.create(
    model=os.getenv("OLLAMA_MODEL", "gpt-3.5-turbo"),  # 로컬 Ollama 모델 사용
    max_tokens=512,
    messages=[{"role": "user", "content": "Hello world"}],
)
print(chat_completion)

# ChatCompletion(id='chatcmpl-sOXJ9vt92vStdAqwOxSASONPSdAUg6yN', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content=' Hello! How can I help you today? ', role='assistant', function_call=None, tool_calls=None))], created=1714527205, model='gpt-3.5-turbo', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=11, prompt_tokens=10, total_tokens=21))
