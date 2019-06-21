## playing around based on the prime factorisation demo

# using the first minor embedding I created 7 * 3 isn't even the most common factorisation of 21! :)
# so now I am trying the precalculated one


import dwavebinarycsp as dbc

# Set an integer to factor
P = 5 * 3 # 7 * 3 #21

# A binary representation of P ("{:06b}" formats for 6-bit binary)
bP = "{:06b}".format(P)
print(bP)



csp = dbc.factories.multiplication_circuit(3)
# Print one of the CSP's constraints, the gates that constitute 3-bit binary multiplication
print(next(iter(csp.constraints)))

# Convert the CSP into BQM bqms
bqm = dbc.stitch(csp, min_classical_gap=.1)
# Print a sample coefficient (one of the programable inputs to a D-Wave system)
print("p0: ", bqm.linear['p0'])


## (just playing in VS Code so not using the helper functions)
# To see helper functions, select Jupyter File Explorer View from the Online Learning page
#from helpers import draw
#draw.circuit_from(bqm)

# Our multiplication_circuit() creates these variables
p_vars = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']

# Convert P from decimal to binary
fixed_variables = dict(zip(reversed(p_vars), "{:06b}".format(P)))
fixed_variables = {var: int(x) for(var, x) in fixed_variables.items()}

# Fix product variables
for var, value in fixed_variables.items():
    bqm.fix_variable(var, value)
    
# Confirm that a P variable has been removed from the BQM, for example, "p0"
print("Variable p0 in BQM: ", 'p0' in bqm)
print("Variable a0 in BQM: ", 'a0' in bqm)


from dwave.system.samplers import DWaveSampler

sampler = DWaveSampler(solver='DW_2000Q_2_1' , token=API_key) #DW_2000Q_5

#sampler = DWaveSampler()

_, target_edgelist, target_adjacency = sampler.structure


# just use pre-calculated embedding (the one I tried to find wasn't very good :())
# Source for this is: https://github.com/dwavesystems/factoring-demo/blob/master/factoring/embedding.py

embedding = embeddings['DW_2000Q_2_1']


from dwave.embedding import embed_bqm, unembed_sampleset

# Minor-Embedding
#import minorminer
# Find an embedding
#embedding = minorminer.find_embedding(bqm.quadratic, target_edgelist)
#if bqm and not embedding:
#    raise ValueError("no embedding found")



# Apply the embedding to the factoring problem to map it to the QPU
bqm_embedded = embed_bqm(bqm, embedding, target_adjacency, 3.0)


# Request num_reads samples
kwargs=dict()
kwargs['num_reads'] = 1000
response = sampler.sample(bqm_embedded, **kwargs)

# Convert back to the problem graph
response = unembed_sampleset(response, embedding, source_bqm=bqm)


from collections import OrderedDict


def to_base_ten(qs):
    def getbits(kyref):
        li = [ky for ky in qs.keys() if ky[0]==kyref and ky[1]<'9']
        v = sum([(qs[ky] * 2**ii) for ii, ky in enumerate(sorted(li))])
        return v
    a = getbits('a')
    b = getbits('b')
    return (a,b)


# Function for converting the response to a dict of integer values
def response_to_dict(response):
    results_dict = OrderedDict()
    for sample, energy in response.data(['sample', 'energy']):
        # Convert A and B from binary to decimal
        a, b = to_base_ten(sample)
        # Aggregate results by unique A and B values (ignoring internal circuit variables)
        if (a, b) not in results_dict:
            results_dict[(a, b)] = energy
            
    return results_dict



results = response_to_dict(response)
[x for x in results if results[x] == 0]



def response_to_engprob(response):
    results_dict = OrderedDict()
    for energy, nocc in response.data(['energy', 'num_occurrences']):
        if energy not in results_dict:
            results_dict[energy] = nocc
        else:
            results_dict[energy] += nocc
            
    return results_dict


results = response_to_engprob(response)

#num = sum(  [ results[x] for x in results] )

def response_energy_array(response):
    
    for sample, energy in response.data(['sample', 'energy']):
        # Convert A and B from binary to decimal
        a, b = to_base_ten(sample)
        # Aggregate results by unique A and B values (ignoring internal circuit variables)
        if (a, b) not in results_dict:
            results_dict[(a, b)] = energy
            
    return results_dict


import matplotlib.pyplot as plt
energy = np.ndarray.flatten(np.array([energy for energy in response.data(['energy']) ]))
max(energy)
plt.hist(energy, normed=True, bins=20)
plt.show()


# probably a built in function for this!
def response_top_samples(response):
     results_dict = OrderedDict()
     for sample, nocc in response.data(['sample', 'num_occurrences']):
          a, b = to_base_ten(sample)
          if (a,b) not in results_dict:
               results_dict[(a, b)] = nocc
          else:
               results_dict[(a, b)]+= nocc
     
     top5 = sorted(results_dict.values(),reverse=True)[0:5]
     print(top5)
     return results_dict
     #for s in top5: print(s, results_dict[s])

response_top_samples(response)





