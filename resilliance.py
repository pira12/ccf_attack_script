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
        if d[0] == "stegosuite" and len(d) == 4:
            d.append(0)
        assert(len(d) == 5)

    data = np.array(data)

    # remove file name from data
    attack_names = data[1:, 1]
    data = data[1:, 2]
    assert(len(attack_names) == len(data))


    attack_names = [attack.replace(name, ("_" + name[-3:]) if "compressed" in attack else "") for attack in attack_names for name in strip_names if name in attack]
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

    plt.title('Successfulness of Attacks')
    plt.xlabel('Attack Names')
    plt.ylabel('Number of unsuccessful attacks')

    plt.show()

plot_data()
