import numpy as np
import random
from copy import deepcopy
import time
from multiprocessing.dummy import Pool as ThreadPool
import itertools


def different_cs(cs1, cs2):
    for i in range(len(cs1)):
        for j in range(len(cs1[0][1])):
            if not(np.isclose(cs1[i][1][j], cs2[i][1][j])):
                return True
    return False


def dist(sample, c):
    return np.linalg.norm(np.subtract(sample[1:], c[1:]))


def eval_group(groups):
    l1 = [l[0] for l in groups[0]]
    l2 = [l[0] for l in groups[1]]

    labels1 = {'AML': l1.count('AML'), 'ALL': l1.count('ALL')}
    labels2 = {'AML': l2.count('AML'), 'ALL': l2.count('ALL')}

    return labels1, labels2


def evalprint_group(group):
    labels = {'AML': 0, 'ALL': 0}
    for sample in group:
        labels[sample[0]] += 1

    print('Número de indivíduos no grupo: ', len(group), '. Sendo ', labels['ALL'], ' ALL e ', labels['AML'],  ' AML.')


def kmeans(samples, k):
    csi = list(np.random.choice(range(len(samples)), size=k*2, replace=False))
    cs = samples[csi[0]], samples[csi[1]]
    prev_cs = samples[csi[2]], samples[csi[3]]
    gs = [[] for i in range(k)]
    while different_cs(cs, prev_cs):
        #start_time = time.time()

        gs = [[] for i in range(k)]
        sums = [np.array([0 for j in range(len(samples[0][1]))]) for i in range(k)]
        counts = [0 for i in range(k)]

        for sample in samples:
            dists = [dist(sample[1], c[1]) for c in cs]
            mindist = dists.index(min(dists))

            gs[mindist].append(sample)
            sums[mindist] = np.add(sums[mindist], sample[1])
            counts[mindist] += 1

        prev_cs = deepcopy(cs)

        for i in range(k):
            cs[i][1] = np.divide(sums[i], counts[i])
        #print(time.time() - start_time)

    return gs


def n2means(samples, mask):
    global indi
    n = 5
    print('Idividual: ', indi)
    indi += 1
    new_samples = []
    for sample in samples:
        new_sample = []
        print(len(mask))
        break
        for j in range(1, len(samples[0])):
            if mask[j] == 1:
                new_sample.append(sample[j])
        new_samples.append([sample[0], np.array(new_sample)])

    best_score = np.inf
    best_model = None
    for i in range(n):
        groups = kmeans(new_samples, 2)

        e0, e1 = eval_group(groups)

        erro = 0

        if e0['ALL'] >= e0['AML']:
            erro += e0['AML']
            erro += e1['ALL']
        else:
            erro += e1['AML']
            erro += e0['ALL']

        if erro <= best_score:
            best_score = erro
            best_model = groups

    return best_score


def crossover_elite(gen, i):
    return gen[i]


def crossover_middle(gen, middle):
    i_parent1, i_parent2 = random.choices(range(int(middle) + 1), k=2)
    parent1 = gen[i_parent1]
    parent2 = gen[i_parent2]

    newborn = [np.random.choice([parent1[i], parent2[i]], p=[0.5, 0.5]) for i in range(len(parent1))]
    return newborn


def crossover_rest(gen):
    return [np.inf] + [random.choice([0,0, 0, 0, 0, 0, 0, 0, 0, 1]) for i in range(len(gen[0]) - 1)]


lines = open('leukemia_big.csv', 'r').read().splitlines()
n_samples = len(lines[0].split(','))

labels = lines.pop(0).split(',')

samples = []

for i in range(n_samples):
    this_line = [labels[i]]
    for line in lines:
        l = line.split(',')
        this_line.append(float(l[i]))
    samples.append(this_line)

n_generations = 100
n_individuals = 50
alpha = 0.002
size_elite = 0.2 * n_individuals
size_middle = 0.3 * n_individuals + size_elite
size_rest = 0.5 * n_individuals + size_middle
n_genes = len(samples[0]) - 1
target_genes = n_genes // 2

gen = [[np.inf] + [random.choice([0, 1]) for i in range(n_genes)] for ind in range(n_individuals)]
errlist = []
genlist = []
sumlist = []

for g in range(n_generations):
    indi = 0
    print('Generation: ', g)
    pool = ThreadPool(4)
    results = pool.starmap(n2means, zip(itertools.repeat(samples), gen))

    break

    for ind in range(n_individuals):
        count = gen[ind][1:].count(1)
        gen[ind][0] = results[ind] + count * alpha
    gen = sorted(gen, key=lambda t: t[0])
    next_gen = []
    for i in range(n_individuals):
        if i <= size_elite:
            next_gen.append(crossover_elite(gen, i))
        elif i <= size_middle:
            next_gen.append(crossover_middle(gen, size_middle))
        else:
            next_gen.append(crossover_rest(gen))
    count = gen[0][1:].count(1)
    print('Melhor indivíduo da geração' , g, '. Erros:', gen[0][0] - count*alpha, ' Com tantos genes: ', count)
    sumlist.append(gen[0][0])
    errlist.append(gen[0][0] - count*alpha)
    genlist.append(count)
    gen = deepcopy(next_gen)

with open('e4-err.txt', 'w') as file:
    for e in errlist:
        print(e, file=file)

with open('e4-gens.txt', 'w') as file:
    for g in genlist:
        print(g, file=file)

with open('e4-tot.txt', 'w') as file:
    for s in sumlist:
        print(s, file=file)