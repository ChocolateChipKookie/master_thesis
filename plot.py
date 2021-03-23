import matplotlib.pyplot as plt


x = []
y = []

file = open("tmp/desmos.log", 'r')
for line in file:
    l = line[1:-2].split(', ')
    y.append(float(l[1]))
    x.append(float(l[0]))

N = 20
current_sum = 0
moving_average = []

for i, y_ in enumerate(y):
    current_sum += y_
    if i >= N:
        current_sum -= y[i-N]
        moving_average.append(current_sum/N)
    else:
        moving_average.append(current_sum/(i+1))


plt.scatter(x, y, marker='.')
plt.plot(x, moving_average, 'r-')

x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,y2))

plt.show()
