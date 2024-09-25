# FLstreamline
FedMCD code for 'A Multi-Model Approach for Handling Concept Drifting Data in Federated Learning"  

## Dependencies
    copy
    math
    time
    numpy
    arff
    torch
    wandb
    csv
    torchvision
    matplotlib
    sklearn
    pandas
    random


## How to run
- When running the artificial dataset, you need to modify the drift_type in the main function, and then determine whether the file path of MyDataset is correct;
- When running the real dataset, just modify the file information of MyDataset;

## Scenario Design
We have established 17 experimental scenarios, varying in 5 temporal drift features (speed, severity, recurrence, frequency, predictability) and 4 spatial drift features(coverage, synchronism, direction, correlation) based on the categorization of conceptdrift in FL. All of the cases generate 1000 batches (i.e. time steps) of data for each of a total of 10 clients. Every batch contains 100 samples.
| No. | Speed       | Severity | Coverage | Other Settings      |
|:---:|:-----------:|:--------:|:--------:|:-------------------:|
| 1   | Abrupt      | High     | High     | Synchronous         |
| 2   | Gradual     | High     | High     | Synchronous         |
| 3   | Incremental | High     | High     | Synchronous         |
| 4   | Abrupt      | Medium   | High     | Synchronous         |
| 5   | Gradual     | Medium   | High     | Synchronous         |
| 6   | Incremental | Medium   | High     | Synchronous         |
| 7   | Abrupt      | Low      | High     | Synchronous         |
| 8   | Gradual     | Low      | High     | Synchronous         |
| 9   | Incremental | Low      | High     | Synchronous         |
| 10  | Abrupt      | High     | Medium   | Synchronous         |
| 11  | Abrupt      | High     | Low      | Synchronous         |
| 12  | Abrupt      | High     | High     | Asynchronous        |
| 13  | Abrupt      | High     | High     | Recurrent           |
| 14  | Abrupt      | High     | High     | Frequent            |
| 15  | Abrupt      | High     | High     | Independent         |
| 16  | Abrupt      | High     | High     | Different Direction |
| 17  | Abrupt      | High     | High     | Unpredictable       |

The detailed settings inside the scenarios are given below:
### Speed: 
describes how fast the current data concept changes to a new concept. We consider three types of speed changes – abrupt, gradual, incremental. Abrupt drift is achieved by suddenly rotating the hyperplane boundary by 180 degrees at time step 200. Gradual drift involves a probabilistic 180-degree change between time steps 200 and 300, where the probability of the old concept decreases by 10% every 10 batches. Starting from time step 300, only the new concept exists. For incremental drift, the data boundary slowly rotates by 180 degrees between time steps 200 and 300. (Scenarios (1, 2, 3), (4, 5, 6), (7, 8, 9))
### Severity: 
describes the degree of concept drift or the distance between the old and new concept. We compare three severity levels – high, medium, low. High severity means rotating the hyperplane by 180 degrees – swapping two labels. Medium and low severity respectively rotate the hyperplane by 120 and 60 degrees. (Scenarios (1, 4, 7), (2, 5, 8), (3, 6, 9))
### Coverage: 
describes how many clients are affected by concept drift around the same time. We consider three coverage levels by changing the number of clients affected by concept drift – 10/10 (high), 5/10 (medium), 1/10 (low). (Scenarios (1, 10, 11))
### Synchronism: 
describes whether concept drift occurs at the same time. This feature reflects the scenarios when a concept drift affects clients in sequence or with some delay. We compare two scenarios – synchronous and asynchronous. In the syn- chronous scenarios, all clients experience concept drift at time step 200 abruptly. In the asynchronous scenarios, the same abrupt drift spreads from clients 1 to 10 at every 100 time steps from time step 100. (Scenarios (1, 12))
### Recurrence: 
describes the cases with returns to previous concepts. Concept drift occurs at time step 100, with a sudden and severe shift in distribution from A to B. At time step 200, the same type of concept drift occurs, with the distribution shifting from B to A. This is referred to as one recurrence. Recurrences also occur at time steps 400 and 700. (Scenario (1, 13))
### Frequency:
describes how often a drift occurs in the data stream. Similar to recurrence, the process where the data distribution shifts from A to B and back to A is referred to as one recurrence. In this scenario, recurrences occur more frequently, starting from time step 50, with a recurrence every 100 time steps. There are a total of 9 recurrences. (Scenarios (13, 14))
### Correlation: 
describes whether and how the drift among clients are correlated. A fully correlated drift occurs when the same type of drift affects all the clients simultaneously (e.g. scenario 1). An independent case is simulated by ten clients’ data occur concept drift with random changes in data distribution occurring at random time steps(Scenario 15).
### Direction: 
describes the changing directions of concept drift among clients. Clients 1-5 experience a sudden drift from concept 1 to concept 2 at time step 200, while clients 6-10 experience a sudden drift from concept 1 to concept 3 at time step 200. (Scenarios (1, 16))
### Predictability: 
describes whether a drift is predictable. Predictability is divided into predictable and unpredictable. Predictable means that concept drift occurs at known time points, while unpredictable means that concept drift occurs at random time points on the client side. The scenario is designed for 10 clients’ data to occur the same type of concept drift at random time steps (scenario 17) to simulate the random case, in comparison to scenarios 12 and 13 which are predictable.

