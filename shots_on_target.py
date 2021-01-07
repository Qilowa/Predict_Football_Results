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
        self.shot_on_target = 0
        self.shot_on_target_allowed = 0
        self.nb_games = 0
        self.goals = 0

        self.x_d = np.linspace(0, 20, 1000)

    def play_match(self, row):
        home_sot = row["HST"]
        away_sot = row["AST"]

        self.shot_on_target += home_sot
        self.shot_on_target_allowed += away_sot
        self.nb_games += 1
        self.goals += row["FTHG"]

    def compute_distributions(self, df):
        as_home = df.loc[df["HomeTeam"] == self.name]
        as_away = df.loc[df["AwayTeam"] == self.name]

        # shots on target
        st1 = np.array(as_home["HST"])
        st2 = np.array(as_away["AST"])
        self.st = np.concatenate((st1, st2), axis=0)

        # shot on target model
        self.kde_st = KernelDensity(bandwidth=1.0, kernel="gaussian")
        self.kde_st.fit(self.st[:, None])


        # shots on target allowed
        sta1 = np.array(as_home["AST"])
        sta2 = np.array(as_away["HST"])

        self.sta = np.concatenate((sta1, sta2), axis=0)

        # shots on target allowed model 
        self.kde_sta = KernelDensity(bandwidth=1.0, kernel="gaussian")
        self.kde_sta.fit(self.sta[:, None])

    def calculate_shot_conversion(self, df):
        df1 = df.loc[df["HomeTeam"] == self.name][["FTHG", "HST"]]
        df1.columns = ["Goals", "ShotOnTarget"]
        df2 = df.loc[df["AwayTeam"] == self.name][["FTAG", "AST"]] 
        df2.columns = ["Goals", "ShotOnTarget"]

        df = pd.concat([df1, df2])

        X, Y = df["ShotOnTarget"], df["Goals"]

        self.slope, self.intercept, r_value, p_value, std_err = stats.linregress(X, Y)


    def predict_goals(self, x):
        return self.slope * x + self.intercept


    def get_kde_SoT(self):
        return self.kde_st

    def get_kde_SoTA(self):
        return self.kde_sta


    def plot(self, mode="st"):
        if mode == "st":
            # score_samples returns the log of the probability density
            logprob = self.kde_st.score_samples(self.x_d[:, None])

            plt.fill_between(self.x_d, np.exp(logprob), alpha=0.5)
            plt.plot(self.st, np.full_like(self.st, -0.01), '|k', markeredgewidth=1) #Return a full array with the same shape and type as a given array.
            plt.ylim(-0.02, 0.22)
        elif mode == "sta":
            logprob = self.kde_sta.score_samples(self.x_d[:, None])

            plt.fill_between(self.x_d, np.exp(logprob), alpha=0.5)
            plt.plot(self.sta, np.full_like(self.sta, -0.01), '|k', markeredgewidth=1) #Return a full array with the same shape and type as a given array.
            plt.ylim(-0.02, 0.22)
        else:
            pass


        plt.show()



def split_data(df, size=0.6):
    n = int(len(df) * size)
    print(n)
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
    dataframe = pd.read_csv("data/F1_2017.csv")

    # split dataframe
    df1, df2 = split_data(dataframe, 0.6)

    dic = calculate_shots_on_target(df1)

    team = dic["Monaco"]

    team.compute_distributions(df1)

    team.get_kde_SoT().sample()

    team.plot()


    """bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=LeaveOneOut())
    grid.fit(team[:, None])

    print(grid.best_params_)"""
