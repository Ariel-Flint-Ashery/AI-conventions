#%%
from huggingface_hub import login, InferenceClient
#login(token = '')
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import random
import networkx as nx
from tqdm import tqdm
import time
#%%
#client = InferenceClient(model="meta-llama/Llama-2-70b-chat-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
API_TOKEN = ''   
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf"
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
#%%%
network_type = 'complete'
iterations=1
r=1000
memory_size = 15
N = 24 #number of agents

def recover_string(answer):
  #template: "A: I will choose the string 'ffj'."
  return answer[-2:-1]

def get_outcome(my_answer, partner_answer):
  if my_answer == partner_answer:
    return 10
  else:
    return -5
  
rules = """You are playing a game repeatedly with another player. The rules of the game are as follows:
1. Your aim in the game is to maximize your own point count. You start with 0 points.
2. In each round, you must choose between Option 0 or Option 1.
3. If you choose Option 0 and the other player chooses Option 0, then you win 10 points. 
4. If you choose Option 0 and the other player chooses Option 1, then you lose 5 points. 
5. If you choose Option 1 and the other player chooses Option 1, then you win 10 points. 
6. If you choose Option 1 and the other player chooses Option 0, then you lose 5 points.
7. Your answer must be of the specific form "I choose Option x.".
8. The minimum possible point tally is 0. If you lose points when your point tally is 0, then the tally remains unchanged."""


def get_llama_response(chat):
    """
    Generate a response from the Llama model.
    """
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    # response = client.text_generation(
    #                                 prompt,
    #                                 do_sample=True,
    #                                 temperature=0.5,
    #                                 top_k=10,
    #                                 max_new_tokens = 20,
    #                                 )
    overloaded = 1
    while overloaded == 1:
      response = query({"inputs": prompt, 
                          "parameters": {"do_sample": True,
                                          "temperature": 0.5,
                                          "top_k":10,
                                          "max_new_tokens": 20,
                                          "return_full_text": False, 
                                          },
                          "options": {"use_cache": False}})
      if type(response)==dict:
        print("AN EXCEPTION")
        time.sleep(20)
      else:
        overloaded=0
    print(response)
    return response[0]['generated_text']

def get_interaction_network(network_type = 'complete', degree=4, rounds = 10, alpha = 0.41, beta=0.54, p=0.5):
  network_dict = {n+1: {'my_history': [], 'partner_history': [], 'interactions': [], 'score': 0, 'score_history': []} for n in range(N)}
  if network_type == 'random_regular':
    graph = nx.random_regular_graph(d=degree, n=len(network_dict.keys()))
    for n in network_dict.keys():
      network_dict[n]['neighbours'] = [i+1 for i in set(graph[n-1])]

  if network_type == 'complete':
    for n in network_dict.keys():
      nodes = list(network_dict.keys())
      nodes.remove(n)
      network_dict[n]['neighbours'] = nodes

  if network_type == 'scale_free':
    graph = nx.scale_free_graph(n=len(network_dict.keys()), alpha=alpha, beta=beta)
    for n in network_dict.keys():
      network_dict[n]['neighbours'] = [i+1 for i in set(graph[n-1])]

  if network_type == 'ER':
    graph = nx.erdos_renyi_graph(n=len(network_dict.keys()), p = p, directed=False)
    for n in network_dict.keys():
      network_dict[n]['neighbours'] = [i+1 for i in set(graph[n-1])]

  return network_dict

def get_prompt(network_dict, p):
  # load player from dictionary
  player = network_dict[p]

  # add initial round
  histories = [{"role": "system", "content": rules},
        ]
  current_score = 0 #local score tracking --ignores global scoring.
  if len(player['my_history']) < memory_size:
    #histories.append({"role": "user", "content": "You are currently playing in round 1. Your point tally is 0. Q: Which Option do you choose, Option 0 or Option 1?"})
    for idx in range(len(player['my_history'])):
      my_answer = player['my_history'][idx] 
      partner_answer = player['partner_history'][idx] 
      outcome = get_outcome(my_answer, partner_answer)
      current_score+=outcome
      if idx != 0:
        histories.append({"role": "assistant", "content": f"I choose Option {my_answer}."})
      
      if outcome > 0: # match
        histories.append({"role": "user", "content": f"In round {idx+1}, you chose Option {my_answer} and the other player chose Option {partner_answer}. Thus you won {outcome} points. You are currently playing in round {idx+2}. Your point tally is {current_score}. Q: Which Option do you choose, Option 0 or Option 1?"})

      if outcome <=0: # no match
        histories.append({"role": "user", "content": f"In round {idx+1}, you chose Option {my_answer} and the other player chose Option {partner_answer}. Thus you lost {outcome} points. You are currently playing in round {idx+2}. Your point tally is {current_score}. Q: Which Option do you choose, Option 0 or Option 1?"})
  
  if len(player['my_history']) >= memory_size:
    indices = list(range(len(player['my_history'])))[-memory_size:]
    for idx, r in enumerate(indices):
      my_answer = player['my_history'][r] 
      partner_answer = player['partner_history'][r] 
      outcome = get_outcome(my_answer, partner_answer)
      current_score+=outcome
      if r != indices[0]:
        histories.append({"role": "assistant", "content": f"I choose Option {my_answer}."})
      if outcome > 0: # match
        histories.append({"role": "user", "content": f"In round {idx+1}, you chose Option {my_answer} and the other player chose Option {partner_answer}. Thus you won {outcome} points. You are currently playing in round {idx+2}. Your point tally is {current_score}. Q: Which Option do you choose, Option 0 or Option 1?"})

      if outcome <=0: # no match
        histories.append({"role": "user", "content": f"In round {idx+1}, you chose Option {my_answer} and the other player chose Option {partner_answer}. Thus you lost {outcome} points. You are currently playing in round {idx+2}. Your point tally is {current_score}. Q: Which Option do you choose, Option 0 or Option 1?"})

  return histories


def update_dict(network_dict, player, my_answer, partner_answer, outcome):
  network_dict[player]['score'] += outcome
  if network_dict[player]['score'] < 0:
    network_dict[player]['score'] = 0 #no negative scores

  network_dict[player]['my_history'].append(my_answer)
  network_dict[player]['partner_history'].append(partner_answer)
  network_dict[player]['score_history'].append(network_dict[player]['score'])

def set_initial_state(network_dict, my_answer, partner_answer):
    outcome = get_outcome(my_answer, partner_answer)
    for p in network_dict.keys():
        if p % 2 == 0:
          update_dict(network_dict, p, my_answer, partner_answer, outcome)
        else:
          update_dict(network_dict, p, partner_answer, my_answer, outcome)

def update_tracker(tracker, p1, p2, p1_answer, p2_answer, outcome):
  tracker['players'].append([p1, p2])
  tracker['answers'].append([p1_answer, p2_answer])
  if outcome > 0:
    tracker['outcome'].append(1)
  else:
    tracker['outcome'].append(0)

def simulation(network_type='complete', rounds=10):
  dataframe = {'simulation': {}, 'tracker': {}}
  interaction_dict = get_interaction_network(network_type = network_type)
  tracker = {'players': [], 'answers': [], 'outcome': []}
  set_initial_state(interaction_dict, '0', '1')
  i=0
  while len(tracker['outcome']) < rounds:
    #randomly choose player and a neighbour
    #print('simulation function called')
    p1 = random.choice(list(interaction_dict.keys()))
    p2 = random.choice(interaction_dict[p1]['neighbours'])
    #add interactions to play history
    interaction_dict[p1]['interactions'].append(p2)
    interaction_dict[p2]['interactions'].append(p1)
    #get prompts
    my_prompt = get_prompt(interaction_dict, p1)
    partner_prompt = get_prompt(interaction_dict, p2)
    #parse through LLM
    my_answer = recover_string(get_llama_response(my_prompt))
    partner_answer = recover_string(get_llama_response(partner_prompt))
    #calculate outcome and update dictionary
    outcome = get_outcome(my_answer, partner_answer)
    update_dict(interaction_dict, p1, my_answer, partner_answer, outcome)
    update_dict(interaction_dict, p2, partner_answer, my_answer, outcome)
    update_tracker(tracker, p1, p2, my_answer, partner_answer, outcome)
    i+=1
    print(f'ITERATION {i}')
    if len(tracker['outcome']) % 20 == 0:
      dataframe['simulation'] = interaction_dict
      dataframe['tracker'] = tracker
      fname = f"70b_LOW_MEMORY_{network_type}_{N}ps_{len(tracker['outcome'])}rds.pkl"
      f = open(fname, 'wb')
      pickle.dump(dataframe, f)
      f.close()
    #time.sleep(2.5)
  return dataframe
#%%
dataframe = simulation(network_type = network_type, rounds = r)
 #%%
# print('STARTING SIMULATIONS')
# start = time.perf_counter()

# for it in range(iterations):
#   #network = {n+1: {'my_history': [], 'partner_history': [], 'interactions': [], 'score': 0, 'score_history': []} for n in range(N)}
#   simulation_dict, tracker_dict = simulation(network_type = network_type, rounds = r)
#   dataframe[it]['simulation'] = simulation_dict
#   dataframe[it]['tracker'] = tracker_dict
#   print(f'SIMULATION {it} COMPLETED')
# print('Time elapsed: %s'% (time.perf_counter()-start))
# fname = f'70b_LOW_MEMORY_{network_type}_{N}ps_{iterations}its_{r}rds.pkl'
# f = open(fname, 'wb')
# pickle.dump(dataframe, f)
# f.close()
# %%
test_chat = [{'role': 'system', 'content': rules},
{'role': 'user',
 'content': "In round 1, you chose Option 0 and the other player chose Option 1. Thus you won 10 points. You are currently playing in round 2. Your point tally is 10. Q: Which Option do you choose, Option 0 or Option 1?"}]

# %%
