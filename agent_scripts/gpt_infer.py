# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import time
client = OpenAI(
    api_key="YOUR API KEY",
    base_url="https://api.chatanywhere.tech/v1"
)  
def infer(prompt, model='gpt-4o-mini'):
    retries = 0
    max_retries = 5
    delay = 10
    if isinstance(prompt, str):
        messages = [
            {"role": "user", "content": prompt}
        ]
    else:
        messages = prompt
    while retries < max_retries:
        try:
            # 发送请求
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False
            )
            
            
            if response:
                break
        
        except Exception as e:
            
            
            # 等待一段时间后重试
            response = ''
            time.sleep(delay)
            retries += 1


    response = response.choices[0].message.content
    return response