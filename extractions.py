import os
import csv
import matplotlib.pyplot as plt
import numpy as np

filename = "./files/successful_extractions_per_cover_image.csv"

def plot_data():
    data = []
    with open(filename, mode='r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            data.append(lines)

    # Extract headers and data
    headers = data[0]
    cover_images = [row[0] for row in data[1:]]
    successful_extractions = [int(row[1]) for row in data[1:]]

    # Plot data
    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(12, 8)
    plt.bar(cover_images, successful_extractions)
    plt.xticks(rotation=45, ha='right')

    plt.title('Successful Extractions Per Cover Image Across All Tools')
    plt.xlabel('Cover Image Type')
    plt.ylabel('Successful Extractions')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()

plot_data()
