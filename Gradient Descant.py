# In this code gradient descant algorithm will be used to train a dataset
# Let's say that we are given a data between cost and length of a pipe.
import numpy as np
import matplotlib.pyplot as plt

pipe_lenght = np.array([100, 150, 200, 250, 300])
pipe_cost = np.array([156, 167, 276, 283, 295])

# Modeling these datas as energy function


plt.plot(pipe_lenght,pipe_cost)

# model function H(x) = theta_0 + theta_1 * x
theta = np.ones(2)
training_number = len(pipe_lenght)

# cost function J(theta) = 1/2m * sum((H(x) - y)^2)
error_func = 1 / (2 * training_number) * ((theta[0] + theta[1] * pipe_lenght - pipe_cost) ** 2)

learning_rate = 0.0001
epsilon = 1e-9
iteration = 0
iter_max = 50000

# gradient descent algorithm
while (np.sum(np.abs(error_func)) > epsilon) and (iter_max > iteration):
    iteration += 1
    for i in range(training_number):
        temp1 = theta[0] - learning_rate * 1 / training_number * (
                    theta[0] + theta[1] * pipe_lenght[i] - pipe_cost[i])
        temp2 = theta[1] - learning_rate * 1 / training_number * (
                    theta[0] + theta[1] * pipe_lenght[i] - pipe_cost[i]) * pipe_lenght[i]

        theta[0] = temp1
        theta[1] = temp2

        error_func[i] = 1 / (2 * training_number) * (theta[0] + theta[1] * pipe_lenght[i] - pipe_cost[i]) ** 2

    print(error_func)

print("Theta:", theta)

plt.plot(pipe_lenght, pipe_cost)
plt.plot(pipe_lenght, theta[0] + theta[1] * pipe_lenght)
plt.show()
