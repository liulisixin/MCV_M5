import matplotlib.pyplot as plt
import json

metrics = []
with open('./output/metrics.json', 'r') as f:
    for line in f:
        metrics.append(line)


iterations = []
losses = []
for metric in metrics:
    splitted_metric = metric.split(',')
    for value in splitted_metric:
        if 'iteration' in value:            
            iterations.append(int(value.split(' ')[-1]))
        elif 'total_loss' in value:
            losses.append(float(value.split(' ')[-1].replace('}\n', '')))

plt.plot(iterations, losses, markevery=5)
plt.title("Model loss")
plt.xlabel("Iteration")
plt.ylabel("Total loss")
plt.savefig("loss.png")