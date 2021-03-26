import matplotlib.pyplot as plt



F = {}

for line in open("tmp/desmos.log", 'r'):
    l = line[1:-2].split(', ')
    F[int(l[0])] = float(l[1])

X = []
Y = []
for x in F:
    X.append(x)
    Y.append(F[x])

X_val = [0]
Y_val = [Y[0]]

for line in open("tmp/val.log", 'r'):
    tmp = line.strip().split()
    X_val.append(int(tmp[0]))
    Y_val.append(float(tmp[1]))

N = 20
current_sum = 0
moving_average = []

for i, y_ in enumerate(Y):
    current_sum += y_
    if i >= N:
        current_sum -= Y[i-N]
        moving_average.append(current_sum/N)
    else:
        moving_average.append(current_sum/(i+1))

plt.figure(figsize=(12, 12))
plt.scatter(X, Y, marker='.')
plt.plot(X, moving_average, 'r-')
plt.plot(X_val, Y_val, 'y-')

x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,y2))

plt.show()
