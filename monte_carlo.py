from team import Team, calculate_shots_on_target, split_data
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

class MonteCarlo():
    def __init__(self, dataframe, iterations=1000):
        self.dataframe = dataframe

        self.train_data, self.test_data = self.__split_data(dataframe, 0.7)

        self.teams = calculate_shots_on_target(dataframe)
        self.iterations = iterations

            
    def get_probability(self, team1, team2):

        home_dist = np.array(self.teams[team1].get_sot_list() + self.teams[team2].get_sota_list()).reshape(-1, 1)
        home_kernel = KernelDensity(bandwidth=1.0, kernel="gaussian")
        home_kernel.fit(home_dist)

        away_dist = np.array(self.teams[team1].get_sota_list() + self.teams[team2].get_sot_list()).reshape(-1, 1)
        away_kernel = KernelDensity(bandwidth=1.0, kernel="gaussian")
        away_kernel.fit(away_dist)

        draw = 0
        home = 0
        away = 0

        x = np.array(self.teams[team1].get_sota_list()).reshape(-1, 1)

        for i in range(self.iterations):
            home_shots = home_kernel.sample()[0][0]
            away_shots = away_kernel.sample()[0][0]

            # print(home_shots, away_shots)
            
            home_goals = np.round(home_shots * self.teams[team1].get_shot_conversion())
            away_goals = np.round(away_shots * self.teams[team2].get_shot_conversion())

            if home_goals == away_goals:
                draw += 1
            elif home_goals > away_goals:
                home += 1
            else:
                away += 1

        
        print(home, draw, away)

        return home/self.iterations, draw/self.iterations, away/self.iterations

    def get_infos(self, team):
        return self.teams[team]

    
    def __split_data(self, df, size=0.6):
        n = int(len(df) * size)
        df1 = df.iloc[:n, :]
        df2 = df.iloc[n:, :]

        return df1, df2

    def make_predictions(self):

        bookmak = []
        model = []
        real = []

        correspondance = {
                0 : "H",
                1 : "D",
                2 : "A"
            }

        for i, row in self.test_data.iterrows():
            home_name = row["HomeTeam"]
            away_name = row["AwayTeam"]
            
            homeW, draw, awayW = self.get_probability(home_name, away_name)

            print(f"{home_name}:{homeW}, Draw:{draw}, {away_name}:{awayW}")

            # get_results
            predicted_prob = max(homeW, draw, awayW)
            if (predicted_prob == homeW):
                predicted_res = "H"
            elif (predicted_prob == draw):
                predicted_res = "D"
            else:
                predicted_res = "A"

            # real result
            result = row["HTR"]

            bookmakers = np.argmin([row["B365H"], row["B365D"], row["B365A"]], axis=0)
            # update results
            real.append(result)
            bookmak.append(correspondance[bookmakers])
            model.append(predicted_res)

            # update training data
            self.train_data.append(row)

            # update home models
            self.teams[home_name].play_match(row)

            # update away models
            self.teams[away_name].play_match(row)

    
        return bookmak, model, real


if __name__ == "__main__":

    dataframe = pd.read_csv("data/Liga/SP1_2019.csv")

    monte = MonteCarlo(dataframe)

    bookmakers, model, real = monte.make_predictions()

    correct = 0
    for m, r in zip(model, real):
        if m == r:
            correct += 1

    print(correct, len(model))

    plt.hist([model, bookmakers, real], label=["Model", "Bookmakers", "Actual"])
    plt.legend()
    plt.show()