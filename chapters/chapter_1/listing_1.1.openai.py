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
    messages=[{"role": "user", "content": "안녕하세요. 한국의 수도는 어디인가요? 한국은 어떤 나라인가요?"}],
)

# 전체 객체(print(chat_completion))는 읽기 어려우므로 필요한 항목만 보기 좋게 출력
choice = chat_completion.choices[0]
usage = chat_completion.usage
# reasoning 모델(DeepSeek-R1 등, Ollama)은 생각 과정을 reasoning 필드로 돌려줌
reasoning = getattr(choice.message, "reasoning", None)

print(f"모델: {chat_completion.model}")
print(f"종료 사유: {choice.finish_reason}")
if reasoning:
    print(f"\n[생각 과정]\n{reasoning.strip()}")
print(f"\n[답변]\n{choice.message.content.strip()}")
print(
    f"\n토큰 사용량: 입력 {usage.prompt_tokens} + 출력 {usage.completion_tokens}"
    f" = 합계 {usage.total_tokens}"
)

# ChatCompletion(id='chatcmpl-sOXJ9vt92vStdAqwOxSASONPSdAUg6yN', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content=' Hello! How can I help you today? ', role='assistant', function_call=None, tool_calls=None))], created=1714527205, model='gpt-3.5-turbo', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=11, prompt_tokens=10, total_tokens=21))
