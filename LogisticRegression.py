import numpy as np
import math

# x = np.array([1, 3, 4, 2, 6, 5, 12, 15, 10, 9])
# y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

dataset = [[1, 1, 0],
            [1, 3, 0],
            [1, 4, 0],
            [1, 2, 0],
            [1, 6, 0],
            [1, 5, 1],
            [1, 12, 1],
            [1, 15, 1],
            [1, 10, 1],
            [1, 9, 1]]

b = [0 for j in range(len(dataset[0])-1)] # List containing θk
b2 = [0 for j in range(len(dataset[0])-1)] # List containing θ'k
learningRate = 0.003
tolerance = 0.001

# Calculate hθ(X)
def hypothesis(dataset, b):
    hyp = b[0]
    for j in range(1, len(b)):
        hyp += b[j] * np.sum(dataset, axis = 0)[j]
    return hyp

# Calculate P(hθ(X))
def probHypothesis(h):
    return 1 / (1 + math.exp(-h))

# Calculate LL(y ∣ X, θ)
def calcLL(dataset, b):
    ll = 0
    for i in range(len(dataset)):
        x = 0
        for j in range(len(b)):
            x += b[j] * dataset[i][j]
        
        ll += (dataset[i][-1] * x) - np.log(1 + math.exp(x))
    
    return ll

# Puts everything together.  I could not get the do/while loop to work,
# so I resorted to just using a for loop
for x in range(100):
    hyp = hypothesis(dataset, b)
    pHyp = probHypothesis(hyp)
    for j in range(len(b2)):
        r = 0
        for i in range(len(dataset)):
            r += (dataset[i][-1] - pHyp) * dataset[i][j] #partial derivative of LL(y ∣ X, θ)
        
        b2[j] = b[j] + (learningRate/len(dataset) * r) # θ'j := θj + α/n * r
    
    llDiff = calcLL(dataset, b) - calcLL(dataset, b2) #Set ΔLL  
    b = b2 # θj := θ'j

# Display expected outcome, the predicted outcome, and the probability of y = 1
for i in range(len(dataset)):
    pred = 1 / (1 + math.exp(-(b[0] + b[1]*dataset[i][1])))
    print(f"x: {dataset[i][1]} - expected: [{dataset[i][2]}] - predicted: [{0 if pred < 0.5 else 1}]- probability: {pred:0.4f}")