from team import Team, calculate_shots_on_target, split_data
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class MonteCarlo():
    def __init__(self, dataframe, iterations=1000):
        self.dataframe = dataframe

        self.train_data, self.test_data = self.__split_data(dataframe, 0.7)

        self.dic = calculate_shots_on_target(dataframe)
        self.iterations = iterations

        self.__compute_all()
    
    def __compute_all(self):
        for k, v in self.dic.items():
            v.compute_distributions(self.train_data)
            print(f"Computing Kernel Density Estimation for {k}")
            
    
    def get_probability(self, team1, team2):

        home_st, home_sta = self.dic[team1].get_kde_shot_on_target(), self.dic[team1].get_kde_shot_on_target_allowed()

        away_st, away_sta = self.dic[team2].get_kde_shot_on_target(), self.dic[team2].get_kde_shot_on_target_allowed()

        draw = 0
        home = 0
        away = 0

        for i in range(self.iterations):
            home_shots = (home_st.sample()[0][0] + away_sta.sample()[0][0] ) / 2
            away_shots = (home_sta.sample()[0][0] + away_st.sample()[0][0] ) / 2
            
            home_goals = np.round(home_shots * self.dic[team1].get_shot_conversion())
            away_goals = np.round(away_shots * self.dic[team2].get_shot_conversion())

            if home_goals == away_goals:
                draw += 1
            elif home_goals > away_goals:
                home += 1
            else:
                away += 1
        
        print(home, draw, away)

        return home/self.iterations, draw/self.iterations, away/self.iterations

    def get_infos(self, team):
        return self.dic[team]

    
    def __split_data(self, df, size=0.6):
        n = int(len(df) * size)
        print(n)
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
            self.dic[home_name].compute_distributions(self.train_data)
            self.dic[home_name].play_match(row)

            # update away models
            self.dic[away_name].compute_distributions(self.train_data)
            self.dic[away_name].play_match(row)

    
        return bookmak, model, real


if __name__ == "__main__":

    dataframe = pd.read_csv("data/Liga/SP1_2019.csv")

    monte = MonteCarlo(dataframe)

    bookmakers, model, real = monte.make_predictions()

    plt.hist([model, bookmakers, real], label=["Model", "Bookmakers", "Real"])
    plt.legend()
    plt.show()