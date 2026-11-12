"""MPI version 2: broadcast of query words and static size-balanced file distribution
(greedy longest-processing-time assignment based on file size)."""
import os
import time
from collections import Counter

from mpi4py import MPI

TOP_N = 10


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")

    comm.Barrier()
    t_total_start = time.perf_counter()

    # Steps 1-3: rank 0 reads the query, lists the corpus files and their sizes
    if rank == 0:
        with open(os.path.join(dataset_dir, "consulta.txt"), "r", encoding="utf-8") as f:
            words = {line.strip().lower() for line in f if line.strip()}
        file_sizes = [
            (name, os.path.getsize(os.path.join(dataset_dir, name)))
            for name in sorted(os.listdir(dataset_dir))
            if name.startswith("file_") and name.endswith(".txt")
            and os.path.isfile(os.path.join(dataset_dir, name))
        ]
        total_files = len(file_sizes)

        # Size-balanced static distribution (LPT): largest files first, each one
        # goes to the process with the smallest accumulated size so far.
        assignments = [[] for _ in range(size)]
        loads = [0] * size
        for name, sz in sorted(file_sizes, key=lambda item: item[1], reverse=True):
            target = loads.index(min(loads))
            assignments[target].append(name)
            loads[target] += sz
    else:
        words = None
        assignments = None
        total_files = None

    # Step 2: broadcast the query words to all processes
    words = comm.bcast(words, root=0)

    # Step 4: each process receives its own static list of files
    my_files = comm.scatter(assignments, root=0)

    # Step 5: local counting, timed per process
    t_local_start = time.perf_counter()
    local_freq = Counter()
    local_tokens = 0
    for name in my_files:
        with open(os.path.join(dataset_dir, name), "r", encoding="utf-8") as f:
            for line in f:
                tokens = [w.lower() for w in line.split()]
                local_tokens += len(tokens)
                for w in tokens:
                    if w in words:
                        local_freq[w] += 1
    local_time = time.perf_counter() - t_local_start

    # Step 6: gather partial results on rank 0
    partials = comm.gather((local_freq, len(my_files), local_time, local_tokens), root=0)

    # Step 7: rank 0 merges results and reports
    if rank == 0:
        total_time = time.perf_counter() - t_total_start

        global_freq = Counter()
        for freq_i, _, _, _ in partials:
            global_freq.update(freq_i)

        total_tokens = sum(p[3] for p in partials)
        total_occurrences = sum(global_freq.values())

        times = [p[2] for p in partials]
        mean_time = sum(times) / len(times)
        load_imbalance = max(times) / mean_time if mean_time > 0 else float("nan")

        print(f"Processes (p): {size}")
        print(f"Files in corpus: {total_files}")
        for i, (_, n_files, t_local, _) in enumerate(partials):
            print(f"  Process {i}: files={n_files}, local_time={t_local:.6f} s")
        print(f"Load imbalance LI = max/mean = {load_imbalance:.4f}")
        print(f"Total execution time: {total_time:.6f} s")
        print(f"Total tokens read: {total_tokens}")
        print(f"Total occurrences found: {total_occurrences}")
        print(f"Top {TOP_N} query words in the corpus:")
        for word, count in global_freq.most_common(TOP_N):
            print(f"  {word}: {count}")


if __name__ == "__main__":
    main()
