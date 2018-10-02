#!/usr/bin/python
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

FILENAME = "../log03.csv"


def plot_random():
    fid = open("../random.csv")
    lines = fid.readlines()
    fid.close()

    points = []
    pareto_front = []

    for line in lines:
        fields = line.rstrip("\n").split(";")

        points.append((np.log(float(fields[1])),float(fields[0])))

    x = []
    y = []

    for point in points:
        x.append(point[0])
        y.append(point[1])

        dominated = False

        for i in range(len(pareto_front) - 1, -1, -1):
            if (pareto_front[i][0] < point[0]) and (pareto_front[i][1] > point[1]): # Dominated
                dominated = True
                break
            elif (point[0] < pareto_front[i][0]) and (point[1] > pareto_front[i][1]): # Dominant
                pareto_front.pop(i)

        if not dominated:
            pareto_front.append(point)

    pareto_x = []
    pareto_y = []

    pareto_front = sorted(pareto_front, key=lambda p: p[0])

    for point in pareto_front:
        pareto_x.append(point[0])
        pareto_y.append(point[1])

    plt.clf()
    plt.scatter(x, y, s=5, c='black', marker='o')
    plt.scatter(pareto_x, pareto_y, s=20, c='blue', marker='o')
    plt.plot(pareto_x, pareto_y, lw=1, c='blue', label='Pareto front')
    plt.xlabel("log(Weights)")
    plt.ylabel("Accuracy")
    plt.legend(loc=0)
    plt.show()
    #plt.savefig("img/random.png")


def plot_grid():
    fid = open("../grid.csv")
    lines = fid.readlines()
    fid.close()

    points = []
    pareto_front = []

    for line in lines:
        fields = line.rstrip("\n").split(";")

        points.append((np.log(float(fields[1])),float(fields[0])))

    x = []
    y = []

    for point in points:
        x.append(point[0])
        y.append(point[1])

        dominated = False

        for i in range(len(pareto_front) - 1, -1, -1):
            if (pareto_front[i][0] < point[0]) and (pareto_front[i][1] > point[1]): # Dominated
                dominated = True
                break
            elif (point[0] < pareto_front[i][0]) and (point[1] > pareto_front[i][1]): # Dominant
                pareto_front.pop(i)

        if not dominated:
            pareto_front.append(point)

    pareto_x = []
    pareto_y = []

    pareto_front = sorted(pareto_front, key=lambda p: p[0])

    for point in pareto_front:
        pareto_x.append(point[0])
        pareto_y.append(point[1])

    plt.clf()
    plt.scatter(x, y, s=5, c='black', marker='o')
    plt.scatter(pareto_x, pareto_y, s=20, c='blue', marker='o')
    plt.plot(pareto_x, pareto_y, lw=1, c='blue', label='Pareto front')
    plt.xlabel("log(Weights)")
    plt.ylabel("Accuracy")
    plt.legend(loc=0)
    plt.show()
    # plt.savefig("img/grid.png")


def plot_generation(generation_number):
    fid = open(FILENAME)
    lines = fid.readlines()
    fid.close()

    points = []
    pareto_front = []

    for line in lines:
        fields = line.rstrip("\n").split(";")

        predicted = int(fields[-1])

        if predicted == 1:
            continue

        if fields[0] == str(generation_number):
            points.append((np.log(float(fields[2])),float(fields[1])))

    x = []
    y = []

    for point in points:
        x.append(point[0])
        y.append(point[1])

        dominated = False

        for i in range(len(pareto_front) - 1, -1, -1):
            if (pareto_front[i][0] < point[0]) and (pareto_front[i][1] > point[1]): # Dominated
                dominated = True
                break
            elif (point[0] < pareto_front[i][0]) and (point[1] > pareto_front[i][1]): # Dominant
                pareto_front.pop(i)

        if not dominated:
            pareto_front.append(point)

    pareto_x = []
    pareto_y = []

    pareto_front = sorted(pareto_front, key=lambda p: p[0])

    for point in pareto_front:
        pareto_x.append(point[0])
        pareto_y.append(point[1])

    plt.clf()
    plt.scatter(x, y, s=5, c='black', marker='o')
    plt.scatter(pareto_x, pareto_y, s=20, c='blue', marker='o')
    plt.plot(pareto_x, pareto_y, lw=1, c='blue', label='Pareto front')
    plt.xlabel("log(Weights)")
    plt.ylabel("Accuracy")
    plt.legend(loc=0)
    plt.show()
    # plt.savefig("img/" + str(generation_number) + ".png")


def plot_summary():
    fid = open(FILENAME)
    lines = fid.readlines()
    fid.close()

    acc = dict()

    for line in lines:
        fields = line.rstrip("\n").split(";")

        key = int(fields[0])
        
        predicted = int(fields[-1])
        
        if predicted == 1:
        	continue

        if key in acc:
            acc[key].append(float(fields[1]))
        else:
            acc[key] = [float(fields[1])]

    min = []
    avg = []
    max = []

    for i in range(51):
        v = acc[i]

        min_acc = 1e10
        max_acc = 0
        avg_acc = 0

        for acc_val in v:
            if acc_val < min_acc:
                min_acc = acc_val

            if acc_val > max_acc:
                max_acc = acc_val

            avg_acc += acc_val

        avg_acc = float(avg_acc) / len(v)

        min.append(min_acc)
        max.append(max_acc)
        avg.append(avg_acc)

    plt.clf()
    plt.plot(min, 'ro-', lw=1, label='MIN')
    plt.plot(max, 'bo-', lw=1, label='MAX')
    plt.plot(avg, 'go-', lw=1, label='AVG')
    plt.grid(color='black', linestyle='--', linewidth=0.1)
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.legend(loc=0)
    plt.show()
    # plt.savefig("img/summary.png")


if __name__ == "__main__":

    #plot_random()

    # plot_grid()

    # plot_generation(50)

    for i in range(51):
        plot_generation(i)

    #plot_summary()

