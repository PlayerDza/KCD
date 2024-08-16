import json
import openai
import numpy as np
from tqdm import tqdm
import pickle
from openai import OpenAI
client = OpenAI(
  api_key=,  # this is also the default, it can be omitted
)

def get_gpt_emb(prompt):

    embedding = client.embeddings.create(
        input=prompt,
        model="text-embedding-ada-002"
    ).data[0].embedding

    return np.array(embedding)

# Read generated profiles
profiles = []
with open('./user/user_prf2.json', 'r') as f:
    for line in f.readlines():
        profiles.append(json.loads(line))

pfx=[]
for i in profiles:
    
    
    pfx.append(i['summarization'])

#print(pfx)
class Colors:
    GREEN = '\033[92m'
    END = '\033[0m'

print(Colors.GREEN + "Encoding Semantic Representation" + Colors.END)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The Profile is:\n" + Colors.END)
print(pfx[0])
print("---------------------------------------------------\n")
emb = get_gpt_emb(pfx[0])
print(Colors.GREEN + "Encoded Semantic Representation Shape:" + Colors.END)
print(emb)

