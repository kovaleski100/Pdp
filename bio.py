import numpy as np
import random
import time
from multiprocessing import Pool, cpu_count
import itertools

def dist(sample, c):
    return np.linalg.norm(sample[1:] - c[1:])

def eval_group(groups):
    labels1 = {'AML': sum(1 for l, _ in groups[0] if l == 'AML'), 'ALL': sum(1 for l, _ in groups[0] if l == 'ALL')}
    labels2 = {'AML': sum(1 for l, _ in groups[1] if l == 'AML'), 'ALL': sum(1 for l, _ in groups[1] if l == 'ALL')}
    return labels1, labels2

def evalprint_group(group):
    labels = {'AML': 0, 'ALL': 0}
    for sample in group:
        labels[sample[0]] += 1
    print('Número de indivíduos no grupo: ', len(group), '. Sendo ', labels['ALL'], ' ALL e ', labels['AML'],  ' AML.')

def kmeans(samples, k, cs):
    prev_cs = np.array(cs)
    gs = [[] for _ in range(k)]

    while not np.allclose(cs, prev_cs):
        gs = [[] for _ in range(k)]
        sums = [np.zeros(len(samples[0][1])) for _ in range(k)]
        counts = np.zeros(k, dtype=int)

        for sample in samples:
            dists = [dist(sample[1], c[1]) for c in cs]
            mindist = np.argmin(dists)

            gs[mindist].append(sample)
            sums[mindist] += sample[1]
            counts[mindist] += 1

        prev_cs = np.copy(cs)

        for i in range(k):
            if counts[i] != 0:
                cs[i][1] = sums[i] / counts[i]

    return gs

def n2means(args):
    samples, mask = args
    n = 5
    new_samples = [[label, np.array([sample[j] for j in range(1, len(sample)) if mask[j] == 1])] for label, sample in samples]

    best_score = np.inf
    best_model = None

    # Larger chunk of work per process
    chunk_size = len(new_samples) // n

    for _ in range(n):
        start_idx = _ * chunk_size
        end_idx = (_ + 1) * chunk_size if _ < n - 1 else len(new_samples)

        # Extract a 1-dimensional array of samples
        flat_samples = np.array([sample[1] for sample in new_samples[start_idx:end_idx]])

        # Generate random indices to choose initial cluster centers
        indices = np.random.choice(len(flat_samples), size=2, replace=False)

        # Use the selected indices to get initial cluster centers
        initial_cs = [flat_samples[i] for i in indices]

        groups = kmeans(new_samples[start_idx:end_idx], 2, cs=initial_cs)

        e0, e1 = eval_group(groups)

        erro = e0['AML'] + e1['ALL'] if e0['ALL'] >= e0['AML'] else e1['AML'] + e0['ALL']

        if erro <= best_score:
            best_score = erro
            best_model = groups

    return best_score

def crossover_elite(gen, i):
    return gen[i]

def crossover_middle(gen, middle):
    i_parent1, i_parent2 = random.choices(range(int(middle) + 1), k=2)
    parent1, parent2 = gen[i_parent1], gen[i_parent2]

    newborn = [np.random.choice([parent1[i], parent2[i]], p=[0.5, 0.5]) for i in range(len(parent1))]
    return newborn

def crossover_rest(gen):
    return [np.inf] + [random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) for _ in range(len(gen[0]) - 1)]

if __name__ == "__main__":
    start = time.time()

    lines = np.genfromtxt('leukemia_big.csv', delimiter=',', dtype=str)
    labels = lines[0, 1:]
    samples = [[lines[i, 0], lines[i, 1:].astype(float)] for i in range(1, len(lines))]

    n_generations = 10
    n_individuals = 50
    alpha = 0.002
    size_elite = int(0.2 * n_individuals)
    size_middle = int(0.3 * n_individuals) + size_elite
    size_rest = int(0.5 * n_individuals) + size_middle
    n_genes = len(samples[0][1])
    target_genes = n_genes // 2

    gen = [[np.inf] + [random.choice([0, 1]) for _ in range(n_genes)] for _ in range(n_individuals)]
    errlist = []
    genlist = []
    sumlist = []

    n_cores = cpu_count()

    for i in range(1, n_cores + 1):
        print("core numbers: ", i)

        for g in range(n_generations):
            print('Generation: ', g)

            pool = Pool(i)
            results = pool.map(n2means, zip(itertools.repeat(samples), gen))
            pool.close()
            pool.join()

            # Update shared variable outside the parallel section
            for ind in range(n_individuals):
                count = np.count_nonzero(gen[ind][1:] == 1)
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

            count = np.count_nonzero(gen[0][1:] == 1)
            sumlist.append(gen[0][0])
            errlist.append(gen[0][0] - count * alpha)
            genlist.append(count)

            gen = np.copy(next_gen)

        with open('e4-err.txt', 'w') as file:
            for e in errlist:
                print(e, file=file)

        with open('e4-gens.txt', 'w') as file:
            for g in genlist:
                print(g, file=file)

        with open('e4-tot.txt', 'w') as file:
            for s in sumlist:
                print(s, file=file)

        print(time.time() - start)
