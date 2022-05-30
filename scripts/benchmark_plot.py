import json
import matplotlib.pyplot as plt
import numpy as np


def plot_data(content):
    nums_threads = []
    times = []
    for benchmark in content["benchmarks"]:
        num_threads = int(benchmark["name"].split("/")[1])
        nums_threads.append(num_threads)
        time = benchmark["cpu_time"]
        times.append(time)
    nums_threads = np.array(nums_threads)
    times = np.array(times)
    times = times / 1e+06

    plt.xlim([0, 7])
    plt.ylim([0, max(times) * 1.2])

    plt.xlabel('threads')
    plt.ylabel('time, ms')

    plt.plot(nums_threads, times)
    plt.savefig("../results/graph.png")


def main():
    file = open('../results/out.json')
    content = ' '.join([l.strip() for l in file.readlines()])
    content = json.loads(content)
    plot_data(content)


if __name__ == "__main__":
    main()
