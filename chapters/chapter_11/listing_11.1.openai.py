from dotenv import load_dotenv
import os
import openai

# 프로젝트 루트의 .env 를 환경변수로 로드
load_dotenv()

client = openai.OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://0.0.0.0:8080/v1"),  # replace with your pi's ip address
    api_key=os.getenv("OPENAI_API_KEY", "1234"),  # replace with your server's api key
)

completion = client.chat.completions.create(
    model=os.getenv("OLLAMA_MODEL", "gpt-3.5-turbo"),
    max_tokens=1024,
    messages=[
        {
            "role": "system",
            "content": "You are Capybara, an AI assistant. Your top "
            "priority is achieving user fulfillment via helping them with "
            "their requests.",
        },
        {
            "role": "user",
            "content": "Building a website can be done in 10 simple steps:",
        },
    ],
)

print(completion.choices[0].message)

# ChatCompletionMessage(content=" 1. Define the purpose and goals of the website.\n2. Choose a domain name and register it.\n3. Select a web hosting service and set up a hosting account.\n4. Install a content management system (CMS) such as WordPress, Joomla, or Drupal.\n5. Choose a template or theme for the website.\n6. Customize the website's design and layout.\n7. Create and publish content, such as text, images, and videos.\n8. Optimize the website for search engines.\n9. Test the website for functionality and usability.\n10. Launch the website and promote it through various channels. ", role='assistant', function_call=None, tool_calls=None)
