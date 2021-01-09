# Predict_Football_Results

## Description

Predict football results using the number of shots on target.
Data are taken from https://www.football-data.co.uk/

### Distributions
The method is to map the distribution of a team's number of shots on target throughout a season and the distributions of the opposite team's number of shots on target they allowed during the season. 
It gives a way to predict how many shots on target will have during a match. The same is done for the opposite team and then mapped into a shot_conversion ratio calculated using the previous matches data. 

### Monte Carlo
Using Monte Carlo simulation, the match is replayed a lot of times to get the probability of the results.

## Example
A match opposing Levante and Granada during the 2019-2020 season.

Let's look at Levante's shots on target throughout the 2019-2020 season and Granada's shots on target allowed.

![histograms of levante and granada](./imgs/distributions.png?raw=true)

Using the Kernel Density Estimation from ```scikit-learn```and combining the two distributions into one that will gives a possible number of shots on target of a team.
The code : 
```
home_dist = np.array(self.teams[team1].get_sot_list() + self.teams[team2].get_sota_list()).reshape(-1, 1)
home_kernel = KernelDensity(bandwidth=1.0, kernel="gaussian")
home_kernel.fit(home_dist)


```

For example, this is the distribution of Levante's number of shots on target (on the right).

![kde models](./imgs/kde.png?raw=true)

Let's do the same for Granada
![kde models](./imgs/distributions2.png?raw=true)

By calling ```home_kernel.sample()```, we get a sample from the distribution, so a number of shots on target. The next step is to multiply this number to a shot conversion rate.
> The shot conversion rate of a team is simply : ```number of goals during the season / number of shots on target during the season```

Using Monte Carlo Simulation : 

```
draw = 0
home = 0     
away = 0

for i in range(self.iterations):
            home_shots = home_kernel.sample()[0][0]
            away_shots = away_kernel.sample()[0][0]
            
            home_goals = np.round(home_shots * self.teams[team1].get_shot_conversion())
            away_goals = np.round(away_shots * self.teams[team2].get_shot_conversion())

            if home_goals == away_goals:
                draw += 1
            elif home_goals > away_goals:
                home += 1
            else:
                away += 1
 ```

## Results
I tested the model on the Spain La Liga 2019-2020 season.
The model is slighty worse than bookmakers prediction (B365).
The model makes good predictions 39% of the time whereas bookmakers are right 41% of the time.

![results](./imgs/results.png?raw=true)
