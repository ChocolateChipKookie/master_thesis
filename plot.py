import matplotlib.pyplot as plt

F = {}

for line in open("unet/data/desmos.log", 'r'):
    l = line[1:-2].split(', ')
    F[int(l[0])] = float(l[1])

X = []
Y = []
for x in F:
    X.append(x)
    Y.append(F[x])

X_val = [0]
Y_val = [Y[0]]

for line in open("unet/data/val.log", 'r'):
    line = line.strip()
    if line == "":
        continue
    tmp = line.split()
    X_val.append(int(tmp[0]))
    Y_val.append(float(tmp[1]))

N = 500
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

trim = 10
if trim:
    X_val = X_val[trim:]
    Y_val = Y_val[trim:]
    trim = trim * 1000
    X = X[trim:]
    Y = Y[trim:]
    moving_average = moving_average[trim:]

plt.scatter(X, Y, marker='.')
plt.plot(X, moving_average, 'r-')
plt.plot(X_val, Y_val, 'y-')

x1,x2,y1,y2 = plt.axis()
if trim:
    plt.axis((x1,x2,y1,y2))
else:
    plt.axis((x1,x2,0,y2))

plt.show()
