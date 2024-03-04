#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#%%
fname = 'LOW_MEMORY_complete_24ps_5its_1000rds_15mem.pkl' #f'pool_pool_size_word_size_N.pkl'
try:
    dataframe = pickle.load(open(fname, 'rb'))
except:
    raise ValueError('NO DATAFILE FOUND')
#%%
def calculate_success_rate(dataframe):
  dataframe = dataframe.copy()
  for i in dataframe.keys():
      for p in dataframe[i]['simulation'].keys():
        dataframe[i]['simulation'][p]['outcome'] = [0]
        for j in range(1, len(dataframe[i]['simulation'][p]['score_history'])):
          if dataframe[i]['simulation'][p]['score_history'][j-1] == dataframe[i]['simulation'][p]['score_history'][j]:
              dataframe[i]['simulation'][p]['outcome'].append(0)
          else:
              dataframe[i]['simulation'][p]['outcome'].append(1)
           
  return dataframe
def plot_score_evolution(dataframe, end_round = 'max'):
  #find average score per round
  #do we want to average over all members, or only the members who participate in the communication?
  dataframe = calculate_success_rate(dataframe)
  fig, axs = plt.subplots(ncols=3, nrows=2, figsize = (15,10))
  axs = axs.flatten()
  #axins = inset_axes(axs[0], width=1.3, height=0.9,  loc=1)
  #axins1 = inset_axes(axs[1], width=1.3, height=0.9, loc=1)
  pop_avg = []
  for i in dataframe.keys():
  #find minimum/maximum number of total population rounds
    print(i)
    minimum = min([len(dataframe[i]['simulation'][p]['score_history']) for p in dataframe[i]['simulation'].keys()])
    end_round = max([len(dataframe[i]['simulation'][p]['score_history']) for p in dataframe[i]['simulation'].keys()])
    print(f'minimum:{minimum}')
    print(f'maximum:{end_round}')
    #shortest path --timestep is when the entire population has played one round
    average_score = []
    std_score = []
    average_success = []
    std_success = []
    for r in range(minimum):
      scores = [dataframe[i]['simulation'][p]['score_history'][r] for p in dataframe[i]['simulation'].keys() if len(dataframe[i]['simulation'][p]['score_history']) == minimum]
      success = [dataframe[i]['simulation'][p]['outcome'][r] for p in dataframe[i]['simulation'].keys() if len(dataframe[i]['simulation'][p]['outcome']) == minimum]
      average_success.append(np.mean(success))
      std_success.append(np.std(success)/np.sqrt(len(success)))
      average_score.append(np.mean(scores))
      std_score.append(np.std(scores)/np.sqrt(len(scores)))
    #print(average_score)
    #print(len(scores))
    print(f'average_success:{average_success}')
    axs[0].errorbar(range(minimum), average_score, yerr = std_score )
    axs[0].set_title(f'Youngest player (n={len(scores)})')
    axs[0].set_ylabel('Average Score')
    axs[0].set_xlabel('Individual Round')

    
    #axins.errorbar(range(minimum), average_success, yerr = std_success)
    axs[3].errorbar(range(minimum), average_success, yerr = std_success)
    axs[3].set_title(f'Youngest player (n={len(scores)})')
    axs[3].set_ylabel('Average Success Rate')
    axs[3].set_xlabel('Individual Round')
    #longest path
    average_score = []
    std_score = []
    average_success = []
    std_success = []
    for r in range(end_round):
      scores = [dataframe[i]['simulation'][p]['score_history'][r] for p in dataframe[i]['simulation'].keys() if len(dataframe[i]['simulation'][p]['score_history']) == end_round]
      success = [dataframe[i]['simulation'][p]['outcome'][r] for p in dataframe[i]['simulation'].keys() if len(dataframe[i]['simulation'][p]['outcome']) == end_round]
      average_success.append(np.mean(success))
      std_success.append(np.std(success)/np.sqrt(len(success)))
      average_score.append(np.mean(scores))
      std_score.append(np.std(scores)/np.sqrt(len(scores)))
    #print(average_score)

    #print(average_success)
    axs[1].errorbar(range(end_round), average_score, yerr = std_score)
    axs[1].set_title(f'Oldest player (n={len(scores)})')
    axs[1].set_ylabel('Average Score')
    axs[1].set_xlabel('Individual Round')

    #axins1.errorbar(range(end_round), average_success, yerr = std_success)
    axs[4].errorbar(range(end_round), average_success, yerr = std_success)
    axs[4].set_title(f'Oldest player (n={len(scores)})')
    axs[4].set_ylabel('Average Success Rate')
    axs[4].set_xlabel('Individual Round')
    #global --timestep follows the longest interaction chain i.e. history of convention
    average_score = []
    std_score = []
    # separate into population rounds
    pop_step_size = 500#int(len(dataframe[i]['simulation'].keys())/2)
    pop_steps = int(len(dataframe[i]['tracker']['outcome'])/pop_step_size)
    left = 0
    for r in range(pop_steps):
      scores = dataframe[i]['tracker']['outcome'][left:left+pop_step_size]
      average_score.append(np.mean(scores))
      std_score.append(np.std(scores)/np.sqrt(len(scores)))
      left+=pop_step_size
    pop_avg.append(average_score)
    # for r in range(rounds):
    #   scores = [dataframe[i][p]['score_history'][r] for p in dataframe[i].keys() if r in dataframe[i][p]['score_history'].keys()]
    #   average_score.append(np.mean(scores))
    #   std_score.append(np.std(scores)/np.sqrt(len(scores)))
    print(average_score)
    axs[2].errorbar(range(pop_steps), average_score)#, yerr = std_score)
    axs[2].set_title('Averaged over population steps')
    axs[2].set_ylabel('Success Rate')
    axs[2].set_xlabel('Population Round')
  lbound = min([len(x) for x in pop_avg])
  data = [x[:lbound] for x in pop_avg]
  axs[5].errorbar(range(lbound), np.mean(data, axis=0), yerr = np.std(data, axis=0)/np.sqrt(len(data)))
  axs[5].set_title('Averaged over population steps (mean over all terms)')
  axs[5].set_ylabel('Success Rate')
  axs[5].set_xlabel('Population Round')
  #for axi in [axins, axins1]:
  #  axi.tick_params(labelleft=False, labelbottom=False)
  plt.tight_layout()
  plt.show()
#%%
plot_score_evolution(dataframe)

# %%
#plot interaction strength

# %%
