import json
import openai
import numpy as np
import os
from tqdm import tqdm


from openai import OpenAI

client = OpenAI(
  api_key=,  # this is also the default, it can be omitted
)

def get_gpt_response_w_system(prompt):
    global system_prompt
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    )
    response = completion.choices[0].message.content
    return response

# read the system_prompt (Instruction) for user profile generation
system_prompt = ""
with open('./user/user_system_prompt.txt', 'r') as f:
    for line in f.readlines():
        system_prompt += line

# read the example prompts of users
example_prompts = []
with open('./user/user_prompts.json', 'r') as f:
    for line in f.readlines():
        u_prompt = json.loads(line)
        example_prompts.append(u_prompt['prompt'])

indexs = len(example_prompts)
picked_id = 0

print(indexs)

class Colors:
    GREEN = '\033[92m'
    END = '\033[0m'

print(Colors.GREEN + "Generating Profile for User" + Colors.END)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The System Prompt (Instruction) is:\n" + Colors.END)
print(system_prompt)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The Input Prompt is:\n" + Colors.END)
print(example_prompts[picked_id])
print("---------------------------------------------------\n")
response = get_gpt_response_w_system(example_prompts[picked_id])
print(Colors.GREEN + "Generated Results:\n" + Colors.END)
print(response)


