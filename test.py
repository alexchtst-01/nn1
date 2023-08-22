import matplotlib.pyplot as plt
import numpy as np
import main

test_input = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [1, 2, 3],
    [1, 3, 10],
    [10, 2, 0],
    [0, 0, 0],
    [3, 4, 2],
    [0, 1, 3]
])

test_output = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1])

model = main.Model_neural(train_itter=100)

pred = model.predict(test_input)

model.Train(X=test_input, Y=test_output, size_Y=4)

error = model.comulative_errros
f = open("error_record.txt", 'w')
f.write(f"the error:\n {error}")
plt.plot(error)
plt.savefig("error-plot.png")
plt.show()