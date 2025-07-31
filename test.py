import os
import praw
from dotenv import load_dotenv

# load_dotenv()

lst = []

with open("outputs/web_datasets/cem_macau_question_set.txt", "r", encoding="utf-8") as f:
    tup = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("**"):
            continue
        tup.append(line)
        if len(tup) >= 2:
            lst.append(tup)
            tup = []
            
    print(len(lst))
    with open("outputs/web_datasets/cem_macau_question_set_2.txt", "w", encoding="utf-8") as f:
        for i in lst:
            f.write(i[0] + "\n" + i[1] + "\n\n")
            

