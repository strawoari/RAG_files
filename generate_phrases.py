import json
import os
import asyncio
import aiofiles
from pathlib import Path
from openai import AsyncAzureOpenAI

with open('chinese_chars.json', 'r', encoding='utf-8') as f:
    data1 = json.load(f)
regions = data1['regions']
age_gender = data1['age_gender']
explanations = data1['explanations']
temperaments = data1['temperaments']
cases = data1['cases']
positions = data1['positions']

with open('data_1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
OUTPUT_DIR = 'chinese_chars'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

generator = AsyncAzureOpenAI(
    api_version=os.getenv('AZURE_API_VERSION'),
    azure_endpoint=os.getenv('AZURE_ENDPOINT'),
    api_key=os.getenv('AZURE_API_KEY')
)

async def get_response(prompt, sem):
    try:
        async with sem:
            response = await generator.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            response = response.choices[0].message.content
        return response
    except Exception as e:
        print(f"Error occurs in get_response: {e}")
        return None

async def handle_one_impersonation(sem, prompt, intent, age_group, education, temperament, language, case, position):
    response = await get_response(prompt, sem)
    if response is None:
        print(f"Failed to get response for intent {intent}")
        return None
    try:
        lst = response.split('\n')
        result = {
            "intent": intent, 
            "age": age_group, 
            "education": education, 
            "temperament": temperament, 
            "language_region": language, 
            "case": case,
            "position": position,
            "phrases": lst
        }
        filename = f'{OUTPUT_DIR}/data_expanded_all_{intent}_3.json'
        async with aiofiles.open(filename, 'a', encoding='utf-8') as f:
            await f.write(json.dumps(result, ensure_ascii=False, indent=4) + "\n")
        print(f"Successfully processed: {intent} - {age_group} - {education} - {temperament} - {language}")
    except Exception as e:
        print(f"Error occurred for intent {intent}: {e}")
        return None

async def main(sem: asyncio.Semaphore):
    tasks = []
    for i in range(len([data[2]])):
        item = data[i]
        intent = item['intent']
        chinese_phrases = item['chinese']
        english_phrases = item['english']
        for position in positions:
            for age_group in age_gender:
                if position == '主驾（开车）' and (age_group == '儿童（4-12 岁）' or age_group == '少年（12-18 岁男性）' or age_group == '少女（12-18 岁女性）'):
                    continue
                if position == '副驾（不开车）' and age_group == '儿童（4-12 岁）':
                    continue
                for education in age_gender[age_group]:
                    for temperament in temperaments:
                        for language in regions:
                            for case in cases[intent]:
                                prompt = f"""你的任务是请模拟一个指定的人的口语，说具有指定意图的语句。

扮演目标人物有以下特征：
你是一个{education}的{age_group}，来自{regions[language]}，习惯用带该区域{language}方言味的普通话。要体现：① {age_group}的常用词汇和话题（如青少年的网络词、老年人的传统表达）；② {education}对应的语言复杂度（如小学毕业可能用词更简单，大学毕业可能更规范）；③ {temperament}的语气。

## 以下是该区域的普通话口语特征，请严格遵循：
{explanations[language]}
请按以上特征模拟：用特色词汇、带方言语法习惯（别用纯方言）。

## 指定意图: {intent} （英文参考）：{english_phrases}

## 正式说法参考：
{chr(10).join(chinese_phrases)}
语义要贴近上面的说法，但得像目标人物说的。

## 场景：结合 {case} 中的具体场景（你在车的 {position} 位）
场景例子：{", ".join(cases[intent][case])}

## 语气要求：{temperament}，具体按 {temperaments [temperament]} 来。

指令（按优先级）：
1. 必须明确提 “{position}” 车窗，并且以符合坐在{position}位的视角逻辑来表达。
2. 先满足意图核心（比如要 “调多少” 就必须说清幅度），再加语气、场景、区域特色。
3. 注重口语化：用词优先选择日常对话中的短句、常用词，避免书面化表达（如不说‘可否帮我’，多用‘能帮我’‘帮我呗’等）
4. 尽量每个要求控制在一个短句内（人跟机器对话，不会说太多）。
5. 至少写 {len (chinese_phrases)} 句。
6. 结合具体场景，让话自然真实。
7. 注意：调高 = 调小，调低 = 调大。
8. 每句句式不一样。（如避免连续用‘请帮我 XX’，可交替用‘XX 一下呗’‘把 XX 调 XX’等不同结构）
9. 只输语句，别加其他。

用以下格式输出：
语句1
语句2
...
"""
                                tasks.append(asyncio.create_task(handle_one_impersonation(sem, prompt, i, age_group, education, temperament, language, case, position)))
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks, return_exceptions=True)
    

if __name__ == "__main__":
    sem = asyncio.Semaphore(5)
    asyncio.run(main(sem))
    
# prompt = f"""Your task is to generate realistic and natural user utterances in both English and Chinese for a specific voice assistant intent.

# ## Intent:
# {intent}

# ## English Sample Utterances:
# {"\n".join(english_phrases)}

# ## Chinese Sample Utterances:
# {"\n".join(chinese_phrases)}

# ## Instructions:
# - Generate 10 natural and diverse user utterances in **English**, and 10 in **Chinese**.
# - Only generate utterances that express the **same intent** and are semantically consistent with the sample utterances.
# - Mix speaking styles: include **direct commands**, **polite requests**, **casual remarks**, and **descriptive statements**.
# - Include meaningful **variations**. For example, if a sample says “close the window halfway”, variants may include “leave the window a bit open” or “关一半”.
# - Avoid robotic, overly formal, or grammatically incorrect phrases.
# - The utterances should sound natural, like something a real driver or passenger would say in everyday conversation.
# - Do not include any intent labels, comments, or explanations—only the utterances.

# ## Output Format (JSON):
# {
#   "english": [
#     "Utterance 1",
#     "Utterance 2",
#     ...
#   ],
#   "chinese": [
#     "Utterance 1",
#     "Utterance 2",
#     ...
#   ]
# }

# ## Example Output:

# Now generate the utterances.
# """