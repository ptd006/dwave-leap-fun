# qubits as coin flips .. 
# started from Ridgeback Network Defense, Inc. coin flip tutorial 
# ref: https://github.com/ridgebacknet/dwave-tutorials

import time
from dwave.system.samplers import DWaveSampler
from neal import SimulatedAnnealingSampler
from dwave.system.composites import EmbeddingComposite

trials = 5000   # How many trials in the coin flipping experiment

useQpu = False   # change this to use a live QPU

if (useQpu):
    sampler = DWaveSampler()
    # We need an embedding composite sampler because not all qubits are
    # working. A trivial embedding lets us avoid dead qubits.
    sampler = EmbeddingComposite(sampler)
else:
    sampler = SimulatedAnnealingSampler()
    
# Initialize a binary quadratic model.
# It will use 2000 qubits. All biases are 0 and all couplings are 0.
bqm = {}       # binary quadratic model

coins = 100
    
for i in range(0, coins):
    bqm[(i,i)] = 0  # indicate a qubit will be used


start = time.time()
response = sampler.sample_qubo(bqm, num_reads=trials)
end = time.time()
total = (end - start)

try:
    qpu_access_time = response.info['timing']['qpu_access_time']
except:
    qpu_access_time = 0
    print('QPU access time is not available. This makes me sad.')


counts = []
for datum in response.data():  # for each series of flips
    n = 0
    for key in datum.sample:   # count how many heads or tails
        if (datum.sample[key] == 1):
            n += 1
    counts += [n]


import matplotlib.pyplot as plt
plt.hist(counts, normed=True, bins=100)
plt.show()

