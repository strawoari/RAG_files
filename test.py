import os
from openai import AzureOpenAI

client = AzureOpenAI(
        api_key=llm_api_key,
        api_version=llm_api_version,
        azure_endpoint=llm_endpoint,
    )

response = client.chat.completions.create(
    stream=True,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model=llm_deployment,
)

for update in response:
    if update.choices:
        print(update.choices[0].delta.content or "", end="")

client.close()