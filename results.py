#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt
#%%
fname = 'fjfjff_24ps_10its_20rds.pkl' #f'pool_pool_size_word_size_N.pkl'
try:
    dataframe = pickle.load(open(fname, 'rb'))
except:
    raise ValueError('NO DATAFILE FOUND')
#%%
def plot_score_evolution(dataframe, rounds, end_round = 'max'):
  #find average score per round
  #do we want to average over all members, or only the members who participate in the communication?
  fig, axs = plt.subplots(ncols=3, figsize = (10,5))
  axs = axs.flatten()
  for i in dataframe.keys():
  #find minimum/maximum number of total population rounds
    minimum = min([len(dataframe[i][p]['score_history'].keys()) for p in dataframe[i].keys()])
    if end_round == 'max':
      end_round = max([len(dataframe[i][p]['score_history'].keys()) for p in dataframe[i].keys()])
    print(minimum)
    #shortest path --timestep is when the entire population has played one round
    average_score = []
    std_score = []
    for r in range(minimum):
      scores = [list(dataframe[i][p]['score_history'].values())[r] for p in dataframe[i].keys()]
      average_score.append(np.mean(scores))
      std_score.append(np.std(scores)/np.sqrt(len(scores)))
    print(average_score)
    axs[0].errorbar(range(minimum), average_score, yerr = std_score)
    axs[0].set_title('Youngest player')
    axs[0].set_ylabel('Average Score')
    axs[0].set_xlabel('Round')

    #longest path
    average_score = []
    std_score = []
    for r in range(end_round):
      scores = [list(dataframe[i][p]['score_history'].values())[r] for p in dataframe[i].keys() if len(dataframe[i][p]['score_history'].values()) == end_round]
      average_score.append(np.mean(scores))
      std_score.append(np.std(scores)/np.sqrt(len(scores)))
    print(average_score)
    axs[1].errorbar(range(end_round), average_score, yerr = std_score)
    axs[1].set_title('Oldest player')
    axs[1].set_ylabel('Average Score')
    axs[1].set_xlabel('Round')

    #global --timestep follows the longest interaction chain i.e. history of convention
    average_score = []
    std_score = []
    for r in range(rounds):
      scores = [dataframe[i][p]['score_history'][r] for p in dataframe[i].keys() if r in dataframe[i][p]['score_history'].keys()]
      average_score.append(np.mean(scores))
      std_score.append(np.std(scores)/np.sqrt(len(scores)))
    print(average_score)
    axs[2].errorbar(range(rounds), average_score, yerr = std_score)
    axs[2].set_title('Averaged over players in global rounds')
    axs[2].set_ylabel('Average Score')
    axs[2].set_xlabel('Round')
  plt.show()
#%%
plot_score_evolution(dataframe, rounds = 20)

# %%
