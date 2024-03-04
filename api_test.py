#%%
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
#%%
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
#%%
API_TOKEN = ''   
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf"
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
#%%
data = query({"inputs": "The answer to the universe is"})
#%%
print(data)
#%%
rules = """You are playing a game repeatedly with another player. The rules of the game are as follows:
1. Your aim in the game is to maximize your own point count. You start with 0 points.
2. In each round, you must choose between Option 0 or Option 1.
3. If you choose Option 0 and the other player chooses Option 0, then you win 10 points. 
4. If you choose Option 0 and the other player chooses Option 1, then you lose 5 points. 
5. If you choose Option 1 and the other player chooses Option 1, then you win 10 points. 
6. If you choose Option 1 and the other player chooses Option 0, then you lose 5 points.
7. Your answer must be of the specific form "I choose Option x.".
8. The minimum possible point tally is 0. If you lose points when your point tally is 0, then the tally remains unchanged."""

test_chat = [{'role': 'system', 'content': rules},
{'role': 'user',
 'content': "In round 1, you chose Option 0 and the other player chose Option 1. Thus you won 10 points. You are currently playing in round 2. Your point tally is 10. Q: Which Option do you choose, Option 0 or Option 1?"}]
prompt = tokenizer.apply_chat_template(test_chat, tokenize=False)

response = query({"inputs": prompt, "parameters": {"do_sample": True, "temperature": 0.5, "top_k":10, "max_new_tokens": 20}, "options": {"use_cache": False}})

"If this works, use"