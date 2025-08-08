import json
import os
import asyncio
import aiofiles
from pathlib import Path
from openai import AsyncAzureOpenAI

with open('english_chars.json', 'r', encoding='utf-8') as f:
    data1 = json.load(f)
regions = data1['regions']
age_gender = data1['age_gender']
temperaments = data1['temperaments']
cases = data1['cases']
positions = data1['positions']

with open('data_1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
generator = AsyncAzureOpenAI(
    api_version=os.getenv('AZURE_API_VERSION'),
    azure_endpoint=os.getenv('AZURE_ENDPOINT'),
    api_key=os.getenv('AZURE_API_KEY')
)
OUTPUT_DIR = 'english_chars'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

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
        filename = f'{OUTPUT_DIR}/data_expanded_all_{intent}.json'
        async with aiofiles.open(filename, 'a', encoding='utf-8') as f:
            await f.write(json.dumps(result, ensure_ascii=False, indent=4) + "\n")
        print(f"Successfully processed: {intent} - {age_group} - {education} - {temperament[0]} - {language}")
    except Exception as e:
        print(f"Error occurred for intent {intent}: {e}")
        return None

async def main(sem: asyncio.Semaphore):
    tasks = []
    for i in range(2, len(data)):
        item = data[i]
        intent = item['intent']
        chinese_phrases = item['chinese']
        english_phrases = item['english']
        for position in positions:
            for age_group in age_gender:
                if position == 'Driver seat' and (age_group == 'child' or age_group == 'teenage boy' or age_group == 'teenage girl'):
                    continue
                if position == 'Front passenger seat' and age_group == 'child':
                    continue
                for education in age_gender[age_group]:
                    for temperament in temperaments:
                        for language in regions:
                            for case in cases[intent]:
                                prompt = f"""Your task is to simulate spoken language from a specific type of person, expressing a specified intent.

The target speaker has the following characteristics:
You are a {language}{age_group} who is {education}, and you usually speak English influenced by the {language} dialect and accent. Your speech should reflect:
① Common words and topics for your age group (e.g., internet slang for teenagers, traditional expressions for the elderly)
② Language complexity appropriate for your education level (e.g., simpler expressions for primary school graduates, more standard/formal usage for university graduates)
③ The tone described by the temperament {temperament}

Below are the regional spoken English features — strictly follow them:
{regions[language]}
Use regional vocabulary and grammatical habits.

Specified Intent: {intent}

Formal English reference: {english_phrases}
For all generated sentences, stay close in meaning to at least one reference sentence, but express it in a way that sounds like how the target person would actually say it.

Scenario: Based on {case}, and you are sitting at the {position} seat in the car
Example situations: {", ".join(cases[intent][case])}

Tone requirement: {temperament}, specifically following the description in '{temperaments[temperament]}'

Instructions (in order of importance):
1. You must clearly mention the “{position}” window, and describe actions from the perspective of someone sitting at that position.
2. First ensure the intent is fulfilled (e.g., if the goal is “adjust amount,” the extent must be specified), then apply tone, scenario, and regional style.
3. Keep it conversational: prioritize short everyday phrases over formal or written ones (e.g., don’t say “Could you please,” say “Can you help me,” “Help me out,” etc.)
4. Keep each request within one short sentence (people don’t say long sentences to voice assistants).
5. Write at least {len(english_phrases)} sentences.
6. Make it natural and realistic by integrating specific scenario details.
7. Note: Raise = close more, lower = open more
8. Use different sentence structures — avoid repeating patterns like always starting with “Can you...”

Only output the spoken lines. Do not include any explanations or extra content.

Output in the following format:
Sentence 1
Sentence 2
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