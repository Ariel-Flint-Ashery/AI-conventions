# -*- coding: utf-8 -*-


import huggingface_hub
huggingface_hub.login(token = '')
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import pickle
import random
import networkx as nx
from tqdm import tqdm
import time
model_dir = 'meta-llama/Llama-2-7b-chat-hf'
model = AutoModelForCausalLM.from_pretrained(model_dir, 
                                             torch_dtype=torch.float16,
                                             device_map='auto',#"balanced_low_0",
                                             #use_flash_attention_2 = True,
                                             )
                                             
tokenizer = AutoTokenizer.from_pretrained(model_dir)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
def format_tokens(dialogs, tokenizer):
    prompt_tokens = []
    for dialog in dialogs:
        if dialog[0]["role"] == "system":
            dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS
                + dialog[0]["content"]
                + E_SYS
                + dialog[1]["content"],
            }
        ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system','user' and 'assistant' roles, "
            "starting with user and alternating (u/a/u/a/u...)"
        )
        """
        Please verify that your tokenizer support adding "[INST]", "[/INST]" to your inputs.
        Here, we are adding it manually.
        """
        dialog_tokens: List[int] = sum(
            [
                tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                    ) + [tokenizer.eos_token_id]
                for prompt, answer in zip(dialog[::2], dialog[1::2])
            ],
            [],
        )
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )
        prompt_tokens.append(dialog_tokens)
    return prompt_tokens

network_type = 'complete'
iterations=5
pool_size = 2
word_size = 2
r=20
N = 24 #number of agents

def get_pool(default_pool = True):
  pool = str()
  while len(set(pool)) <= 1:
    if default_pool == True:
      # pool = ''.join(random.choices('fj', k = pool_size))
      pool = 'fj'
    else:
      #pool = ''.join(random.choices(alphabet, k = pool_size))
      pool = ''.join(random.choices('fj', k = pool_size))
  return pool

def recover_string(answer):
  #template: "A: I will choose the string 'ffj'."
  return answer[-word_size:]

def get_outcome(my_answer, partner_answer):
  if my_answer == partner_answer:
    return 10
  else:
    return -5

pool = get_pool(default_pool = True)
#print("Players can choose $s characters from the pool: %s" % (word_size, pool))
correct_example = ''.join(random.choices(pool, k = word_size))
incorrect_example = ''.join(random.choices(pool, k = word_size))
while incorrect_example == correct_example:
  incorrect_example = ''.join(random.choices(pool, k = word_size))

rules = f"""You are playing a game repeatedly with another player. In this game, you are presented with a pool of letters: '{pool}'.
The rules of the game are as follows:
1. You must produce a two character string from the pool.
2. If you and the other player produce matching strings, then you win 10 points. For example: "In round X, you produced '{correct_example}' and the other player produced '{correct_example}'. Thus you won 10 points."
3. If you produce a string that does not match the other player's string, then you lose 5 points. For example: "In round X, you produced '{correct_example}' and the other player produced '{incorrect_example}'. Thus you lost 5 points."
4. Your aim is to maximise your own point count.
5. Your answer must be one word indicating the chosen string. For example: "'XX'".
6. The minimum possible point tally is 0. If you lose points when your tally is 0, then the tally remains unchanged."""

print(rules)

def get_llama_response(prompt):
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        The model's response.
    """

    #GENERATE METHOD
    chat = format_tokens([prompt], tokenizer)
    tokens = torch.tensor(chat).long()
    tokens = tokens.to("cuda:0")
    #model_input = tokenizer(prompt, return_tensors=“pt”).to("cuda")
    #model.eval()
    outputs = model.generate(
                input_ids=tokens,
                do_sample=True,
                #top_p=top_p,
                temperature=0.5,
                top_k=6,
                max_new_tokens = 5,
            )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

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
        {"role": "user", "content": "You are currently playing in round 1. Your point tally is 0. Q: What two character string would you like to produce from the available pool?"},
        ]

  for idx in range(len(player['my_history'])):
    my_answer = player['my_history'][idx] #r
    partner_answer = player['partner_history'][idx] #r
    outcome = get_outcome(my_answer, partner_answer)
    histories.append({"role": "assistant", "content": f"'{my_answer}'"})
    if outcome > 0: # match
      histories.append({"role": "user", "content": f"In round {idx+1}, you produced '{my_answer}' and the other player produced '{partner_answer}'. Thus you won {outcome} points. You are currently playing in round {idx+2}. Your point tally is {player['score_history'][idx]}. Q: What two character string would you like to produce from the available pool?"})

    else: # no match
      histories.append({"role": "user", "content": f"In round {idx+1}, you produced '{my_answer}' and the other player produced '{partner_answer}'. Thus you lost {outcome} points. You are currently playing in round {idx+2}. Your point tally is {player['score_history'][idx]}. Q: What two character string would you like to produce from the available pool?"})

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
  tracker['p1'].append(p1_answer)
  tracker['p2'].append(p2_answer)
  if outcome > 0:
    tracker['outcome'].append(1)
  else:
    tracker['outcome'].append(0)

def simulation(network_type='complete', rounds=10):
  interaction_dict = get_interaction_network(network_type = network_type)
  tracker = {'players': [], 'p1': [], 'p2': [], 'outcome': []}
  set_initial_state(interaction_dict, 'ff', 'jj')
  while min([len(interaction_dict[p]['interactions']) for p in interaction_dict.keys()])< rounds-1:
    #randomly choose player and a neighbour
    #print('simulation function called')
    p1 = random.choice(list(interaction_dict.keys()))
    while len(interaction_dict[p1]['my_history'])>39:
        p1 = random.choice(list(interaction_dict.keys()))
    p2 = random.choice(interaction_dict[p1]['neighbours'])
    while len(interaction_dict[p2]['my_history'])>39:
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

  return interaction_dict, tracker

"""**RUNNING THE SIMULATION**

We are now ready to test the simulation! Let us see how it works.
"""
print('STARTING SIMULATIONS')
start = time.perf_counter()
dataframe = {it: {'simulation': {}, 'tracker': {}} for it in range(iterations)}
for it in range(iterations):
  #network = {n+1: {'my_history': [], 'partner_history': [], 'interactions': [], 'score': 0, 'score_history': []} for n in range(N)}
  simulation_dict, tracker_dict = simulation(network_type = network_type, rounds = 20)
  dataframe[it]['simulation'] = simulation_dict
  dataframe[it]['tracker'] = tracker_dict
  print(f'SIMULATION {it} COMPLETED')
print('Time elapsed: %s'% (time.perf_counter()-start))
fname = f'{network_type}_{pool}_{N}ps_{iterations}its_{r}rds_{word_size}words.pkl'
f = open(fname, 'wb')
pickle.dump(dataframe, f)
f.close()

