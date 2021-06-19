import matplotlib.pyplot as plt


def moving_average(vals, n):
    current_sum = 0
    ma = []

    for i, y in enumerate(vals):
        current_sum += y
        if i >= n:
            current_sum -= vals[i - n]
            ma.append(current_sum / n)
        else:
            ma.append(current_sum / (i + 1))
    return ma


def plot_colorful():
    F = {}

    for line in open("colorful/data/desmos.log", 'r'):
        l = line[1:-2].split(', ')
        F[int(l[0])] = float(l[1])

    X = []
    Y = []
    for x in F:
        X.append(x)
        Y.append(F[x])

    X_val = [0]
    Y_val = [Y[0]]

    for line in open("colorful/data/val.log", 'r'):
        line = line.strip()
        if line == "":
            continue
        tmp = line.split()
        X_val.append(int(tmp[0]))
        Y_val.append(float(tmp[1]))

    ma = moving_average(Y, 500)

    plt.figure(figsize=(12, 12))

    trim = 1
    if trim:
        X_val = X_val[trim:]
        Y_val = Y_val[trim:]
        trim = trim * 5000
        X = X[trim:]
        Y = Y[trim:]
        ma = ma[trim:]

    plt.scatter(X, Y, marker='.')
    plt.plot(X, ma, 'r-')
    plt.plot(X_val, Y_val, 'y-')

    x1, x2, y1, y2 = plt.axis()
    if trim:
        plt.axis((x1, x2, y1, y2))
    else:
        plt.axis((x1, x2, 0, y2))

    plt.show()


def plot_gan():
    F = {}

    for line in open("gan/data/loss.log", 'r'):
        line = line.strip()
        if line == "":
            continue
        l = line[1:-1].split(', ')
        i = int(l[0])
        vals = tuple(float(x) for x in l[1:])
        F[i] = vals

    X = []
    Y = []
    for x in sorted(F):
        X.append(x)
        Y.append(F[x])

    n = 500
    moving_averages = []
    for i in range(len(Y[0])):
        y_ = [y[i] for y in Y]
        moving_averages.append(moving_average(y_, n))

    X_val = [0]
    Y_val = [Y[0]]

    for line in open("gan/data/val.log", 'r'):
        line = line.strip()
        if line == "":
            continue
        tmp = line.split()
        X_val.append(int(tmp[0]))
        Y_val.append(tuple(float(x) for x in tmp[1:]))

    trim = 1
    val_period = X_val[1]
    if trim:
        X_val = X_val[trim:]
        Y_val = Y_val[trim:]
        trim = trim * val_period
        X = X[trim:]
        Y = Y[trim:]
        for i in range(len(moving_averages)):
            moving_averages[i] = moving_averages[i][trim:]

    fig, axs = plt.subplots(2)
    plt.figure(figsize=(12, 6))

    # Loss plot:
    # Scatter points
    # Draw running mean
    # Draw validation results
    fig_loss = axs[1]

    losses = [y[4] for y in Y]
    loss_ma = moving_averages[4]
    loss_vals = [y[4] for y in Y_val]

    fig_loss.scatter(X, losses, marker='.')
    fig_loss.plot(X, loss_ma, 'r-')
    fig_loss.plot(X_val, loss_vals, 'y-')

    # D fake (red) and real (blue)
    # G fake (green)
    # Moving averages are full, validation is striped
    gan_game = axs[0]
#    gan_game.set_yscale("log")

    g_fake_ma = moving_averages[5]
    g_fake_val = [y[5] for y in Y_val]
    d_real_ma = moving_averages[1]
    d_real_val = [y[1] for y in Y_val]
    d_fake_ma = moving_averages[2]
    d_fake_val = [y[2] for y in Y_val]

    gan_game.plot(X, g_fake_ma, '-', color=(0.0, 1.0, 0.0))
    gan_game.plot(X_val, g_fake_val, '-', color=(0.0, 0.5, 0.0))
    gan_game.plot(X, d_real_ma, '-', color=(0.0, 0.0, 1.0))
    gan_game.plot(X_val, d_real_val, '-', color=(0.0, 0.0, 0.5))
    gan_game.plot(X, d_fake_ma, '-', color=(1.0, 0.0, 0.0))
    gan_game.plot(X_val, d_fake_val, '-', color=(0.5, 0.0, 0.0))

    plt.show()

if __name__ == "__main__":
    plot_colorful()


