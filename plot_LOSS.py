import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('saved_dictionary_1.pkl', 'rb') as f:
  loaded = pickle.load(f)
#with open('saved_dictionary_flash_nonperf_9.pkl', 'rb') as f:
#  loaded_nonperf = pickle.load(f)
#with open('saved_dictionary_flash_perf_1.pkl', 'rb') as f:
with open('saved_dictionary_rebuild_1.pkl', 'rb') as f:
  loaded_perf = pickle.load(f)

size = len(loaded['loss_list'])
START = 30
ep = np.arange(START,size)

loss = np.zeros(size-START)
#loss_nonperf = np.zeros(size-START)
loss_perf = np.zeros(size-START)
for i in range(size-START):
  loss[i] = loaded['loss_list'][i+START]
#  loss_nonperf[i] = loaded_nonperf['loss_list'][i+START]
  loss_perf[i] = loaded_perf['loss_list'][i+START]

EP = np.arange(START)
LOSS = np.zeros(START)
#LOSS_nonperf = np.zeros(START)
LOSS_perf = np.zeros(START)
for i in range(START):
  LOSS[i] = loaded['loss_list'][i]
#  LOSS_nonperf[i] = loaded_nonperf['loss_list'][i]
  LOSS_perf[i] = loaded_perf['loss_list'][i]

fig, ax = plt.subplots()
#ax.set_yscale('log', base=2)
#plt.yscale('log')
ax.plot(ep, loss, label='Standard')
#ax.plot(ep, loss_nonperf, label='Flash NonPerf')
ax.plot(ep, loss_perf, label='Flash Perf')
ax.set(xlabel='Iterations', ylabel='Loss',
       title='Loss Curves')
plt.legend(loc="lower right")

fig.savefig("Flash-Attention-2_act.png")

plt.cla()
plt.clf()
plt.close()
fig, ax = plt.subplots()
#ax.set_yscale('log', base=2)
#plt.yscale('log')
ax.plot(EP, LOSS, label='Standard')
#ax.plot(EP, LOSS_nonperf, label='Flash NonPerf')
ax.plot(EP, LOSS_perf, label='Flash Perf')
ax.set(xlabel='Iterations', ylabel='Loss',
       title='Loss Curves')
plt.legend(loc="upper right")

fig.savefig("Flash-Attention-1_rebuild.png")
