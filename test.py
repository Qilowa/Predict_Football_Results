import numpy as np

def brier_score(prob_h, prob_d, prob_a, result):
    if result == "H":
        vector = [1, 0, 0]
    elif result == "D":
        vector = [0, 1, 0]
    else:
        vector = [0, 0, 1]

    return (pow((prob_h - vector[0]), 2) + pow((prob_d - vector[1]), 2) + pow((prob_a - vector[2]), 2))/3



        """x_d = np.linspace(-10, 20, 1000)

        x = np.array(self.teams[team1].get_sot_list()).reshape(-1, 1)
        y = np.array(self.teams[team2].get_sota_list()).reshape(-1, 1)

        kernel = KernelDensity(bandwidth=1.0, kernel="gaussian")
        kernel.fit(x)

        kernel2 = KernelDensity(bandwidth=1.0, kernel="gaussian")
        kernel2.fit(y)

        logprob = kernel.score_samples(x_d[:, None])
        logprob2 = kernel2.score_samples(x_d[:, None])

        with plt.style.context('ggplot'):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            
            ax1.set_title("Histogram of Levante's shot on target")
            ax1.hist(x, density=True, bins=15)

            ax2.set_title("Histogram of Granada's shot on target allowed")
            ax2.hist(y, color="blue", density=True, bins=15)
            ax2.grid(True)

        plt.show()"""