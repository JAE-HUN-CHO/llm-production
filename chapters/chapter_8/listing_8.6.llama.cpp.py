from dotenv import load_dotenv
import os
import time
from llama_cpp import Llama

# 프로젝트 루트의 .env 를 환경변수로 로드
load_dotenv()

# 모델 경로는 LLAMA_MODEL_PATH 로 변경 가능 (예: 작은 TinyLlama GGUF)
MODEL_PATH = os.getenv(
    "LLAMA_MODEL_PATH", "./models/wizardcoder-python-7b-v1.0.Q2_K.gguf"
)
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=int(os.getenv("N_CTX", "2048")),
    n_threads=int(os.getenv("N_THREADS", "0")) or None,
)

start_time = time.time()
output = llm(
    "Q: Write python code to reverse a linked list. A: ",
    max_tokens=int(os.getenv("MAX_TOKENS", "200")),
    stop=["Q:"],
    echo=True,
)
end_time = time.time()

print(output["choices"])
# [
#     {'text': "Q: Write python code to reverse a linked list. A:
#         class Node(object):
#             def __init__(self, data=None):
#                 self.data = data
#                 self.next = None

#         def reverse_list(head):
#             prev = None
#             current = head
#             while current is not None:
#                 next = current.next
#                 current.next = prev
#                 prev = current
#                 current = next
#             return prev

#         # example usage
#         # initial list
#         head = Node('a')
#         head.next=Node('b')
#         head.next.next=Node('c')
#         head.next.next.next=Node('d')
#         print(head)
#         reverse_list(head) # call the function
#         print(head)

#         # expected output: d->c->b->a",
#     'index': 0,
#     'logprobs': None,
#     'finish_reason': 'stop'
#     }
# ]

print(f"Elapsed time: {end_time - start_time:.3f} seconds")
# Elapsed time: 239.457 seconds
