import numpy as np

def brier_score(prob_h, prob_d, prob_a, result):
    if result == "H":
        vector = [1, 0, 0]
    elif result == "D":
        vector = [0, 1, 0]
    else:
        vector = [0, 0, 1]

    return (pow((prob_h - vector[0]), 2) + pow((prob_d - vector[1]), 2) + pow((prob_a - vector[2]), 2))/3