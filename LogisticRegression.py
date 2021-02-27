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
learningRate = 0.003 # Learning Rate α 
tolerance = 0.001 # Error value ϵ

# Calculate hθ(X)
def hypothesis(dataset, b):
    hyp = b[0]
    for j in range(1, len(b)):
        hyp += b[j] * np.sum(dataset, axis = 0)[j]
    return hyp

# Calculate P(hθ(X))
def probHypothesis(h):
    return 1 / (1 + math.exp(-h))

# Calculate LL(y | X, θ)
def calcLL(dataset, b):
    ll = 0
    for i in range(len(dataset)):
        x = 0
        for j in range(len(b)):
            x += b[j] * dataset[i][j]
        
        ll += (dataset[i][-1] * x) - np.log(1 + math.exp(x))
    
    return ll

# Puts everything together.  Something in relation to LL is not working, so I am using a for loop to iterate.
# The issue is that, somewhere, somehow, at iteration 3 and beyond θ and θ' are always the same value at the point
# where LL is calculated.  This causes ΔLL = 0, making it useless to compare to ϵ.
for x in range(100):
#while True:
    hyp = hypothesis(dataset, b)
    pHyp = probHypothesis(hyp)
    for j in range(len(b2)):
        r = 0
        for i in range(len(dataset)):
            r += (dataset[i][-1] - pHyp) * dataset[i][j] # Partial derivative of LL(y | X, θ)
        
        b2[j] = b[j] + (learningRate/len(dataset) * r) # θ'j := θj + α/n * r
        print(learningRate/len(dataset) * r)
    
    llDiff = calcLL(dataset, b2) - calcLL(dataset, b)  # ΔLL := LL(y | X, θ') - LL(y | X, θ)
    b = b2 # θj := θ'j
#    if llDiff < 0.001:
#        break

# Display expected outcome, the predicted outcome, and the probability of y = 1
for i in range(len(dataset)):
    pred = 1 / (1 + math.exp(-(b[0] + b[1]*dataset[i][1])))
    print(f"x: {dataset[i][1]} - expected: [{dataset[i][2]}] - predicted: [{0 if pred < 0.5 else 1}]- probability: {pred:0.4f}")
