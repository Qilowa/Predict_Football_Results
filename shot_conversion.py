import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from team import Team, calculate_shots_on_target, split_data
from scipy import stats

def create_data(df, team):
    df1 = df.loc[df["HomeTeam"] == team][["FTHG", "HST"]]
    df1.columns = ["Goals", "ShotOnTarget"]
    df2 = df.loc[df["AwayTeam"] == team][["FTAG", "AST"]] 
    df2.columns = ["Goals", "ShotOnTarget"]

    return pd.concat([df1, df2])


dataframe1 = pd.read_csv("data/Ligue_1/F1_2017.csv")
dataframe2 = pd.read_csv("data/Ligue_1/F1_2018.csv")

dataframe = pd.concat([dataframe1, dataframe2])


# split dataframe
df1, df2 = split_data(dataframe, 0.8)

#team = create_data(df1, "Marseille")
team = df1[["FTAG", "AST"]]
team.columns = ["Goals", "ShotOnTarget"]

X, Y = team["ShotOnTarget"], team["Goals"]

slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

def predict(x):
    return slope * x + intercept

line = predict(X)

#plt.scatter(X, Y)
#plt.plot(X, line, c='r')

sns.jointplot(x="ShotOnTarget", y="Goals", data=team, kind="reg")
plt.grid(True)

plt.show()