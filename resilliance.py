import os
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np


filename = "./files/extraction_summary.csv"

names = ["_F-16C", "_roswell_company_secrets", "_noisy_satellite_image", "_ripsawm5"]
strip_names = [name + ext for name in names for ext in [".png", ".jpg"]]

def plot_data():
    data = []
    with open(filename, mode='r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            data.append(lines)

    data[0].append("levenshtein_distance")
    for d in data:
        if len(d) != 5:
            d.append(0)

    data = np.array(data)

    # remove file name from data
    attack_names = data[1:, 1]
    data = data[2:, 2]


    attack_names = [attack.replace(name, "") for attack in attack_names for name in strip_names if name in attack]
    data = [int(d) for d in data]


    # plot data


    # attack names on x-axis
    counter = dict()
    for name, data in zip(attack_names, data):
        if name in counter:
            counter[name] += data
        else:
            counter[name] = 0


    x = counter.keys()
    y = counter.values()

    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(12, 8)
    plt.bar(x, y)
    plt.xticks(rotation=45, ha='right')

    plt.title('Resilience of Attacks.')
    plt.xlabel('Attack Names')
    plt.ylabel('Resilience')

    plt.show()

plot_data()
