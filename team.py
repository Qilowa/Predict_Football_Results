import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity
import numpy as np
from scipy import stats

class Team():
    def __init__(self, name):
        self.name = name
        self.nb_games = 0
        self.goals = 0

        self.shot_on_target_list = []
        self.shot_on_target_allowed_list = []

        self.x_d = np.linspace(0, 20, 1000)

    def play_match(self, row):

        home_sot = row["HST"]
        away_sot = row["AST"]

        if row["HomeTeam"] == self.name:
            self.nb_games += 1
            self.goals += row["FTHG"]

            self.shot_on_target_list.append(home_sot)
            self.shot_on_target_allowed_list.append(away_sot)
        else:
            self.nb_games += 1
            self.goals += row["FTHG"]

            self.shot_on_target_list.append(away_sot)
            self.shot_on_target_allowed_list.append(home_sot)

    def get_sot_list(self):
        return self.shot_on_target_list

    def get_sota_list(self):
        return self.shot_on_target_allowed_list


    def get_shot_conversion(self):
        return self.goals/sum(self.shot_on_target_list)



def split_data(df, size=0.6):
    n = int(len(df) * size)
    df1 = df.iloc[:n, :]
    df2 = df.iloc[n:, :]

    return df1, df2



def calculate_shots_on_target(df) -> dict:
    dic = dict()

    for i, row in df.iterrows():
        home_name = row["HomeTeam"]
        away_name = row["AwayTeam"]

        if home_name not in dic.keys():
            dic[home_name] = Team(home_name)

        dic[home_name].play_match(row)
        
        
        if away_name not in dic.keys():
            dic[away_name] = Team(away_name)

        dic[away_name].play_match(row)


    return dic 


if __name__ == "__main__":
    dataframe = pd.read_csv("data/Liga/SP1_2019.csv")

    # split dataframe
    df1, df2 = split_data(dataframe, 0.7)

    dic = calculate_shots_on_target(df1)

    # team 1
    team1 = dic["Sociedad"]

    team1.compute_distributions(df1)

    #team1.plot(mode="st")
    team1.plot(mode="sta")

    #team 2
    team2 = dic["Barcelona"]

    team2.compute_distributions(df1)

    #team2.plot(mode="sta")
    team2.plot(mode="st")

    plt.show()


    """bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=LeaveOneOut())
    grid.fit(team[:, None])

    print(grid.best_params_)"""
