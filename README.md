# Predict_Football_Results

## Description

Predict football results using the number of shots on target.

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
![kde models](./imgs/kde2.png?raw=true)
