import matplotlib.pyplot as plt
import numpy as np
import json
import glob

def createPie(file_name):
    data = json.load(open(file_name))

    x = [
        int(data['alma-prox']/data['image-number'] * 100),
        int(data['dag-0.001']/data['image-number'] * 100),
        int(data['dag-0.003']/data['image-number'] * 100),
        int(data['pdpgd']/data['image-number'] * 100)
    ]

    l = [
        "alma-prox",
        "dag-0.001",
        "dag-0.003",
        "pdpgd"
    ]

    l_ = []
    x_ = []

    for i in range(len(x)):
        if(x[i] != 0):
            x_.append(x[i])
            l_.append(l[i])

    plt.pie(x_, labels = l_, startangle = 90)
    plt.show()

def createBar(file_name):
    print(file_name)
    data = json.load(open(file_name))

    x = [
        int(data['alma-prox']/data['image-number'] * 100),
        int(data['dag-0.001']/data['image-number'] * 100),
        int(data['dag-0.003']/data['image-number'] * 100),
        int(data['pdpgd']/data['image-number'] * 100),
    ]

    l = [
        "ALMA-PROX",
        "DAG-0.001",
        "DAG-0.003",
        "PDPGD"
    ]

    print(data["name"] + " , PE > ")
    plt.rcParams.update({'font.size': 10})
    plt.title(data["name"] + " , PE > " + str(int(data["apsr"] * 100)) + "%")

    plt.ylabel('Percentage of validation set')
    plt.yticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"])

    plt.ylim(0, 100) 

    plt.bar(l, x, color=['red', 'green', 'lime', 'blue'])
    plt.savefig("./Images/" + data["name"] + "_APSR_" + str(int(data["apsr"] * 100)) + ".png", bbox_inches="tight")
    plt.clf()

l_ = glob.glob("./BestAttack/*")

for l in l_:
    createBar(l)