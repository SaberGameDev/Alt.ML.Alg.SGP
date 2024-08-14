import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVR

X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)

lw = 2

fig, ax = plt.subplots()
ax.plot(
        X,
        svr.fit(X, y).predict(X),
        color="red",
       # lw=lw,
        label="model",
    )
ax.scatter(
        X[svr.support_],
        y[svr.support_],
        facecolor="none",
        edgecolor="red",
        s=50,
        label="RBF support vectors",
    )
ax.scatter(
        X[np.setdiff1d(np.arange(len(X)), svr.support_)],
        y[np.setdiff1d(np.arange(len(X)), svr.support_)],
        facecolor="none",
        edgecolor="k",
        s=50,
        label="other training data",
    )
ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=1,
        fancybox=True,
        shadow=True,
    )

fig.text(0.5, 0.04, "data", ha="center", va="center")
fig.text(0.06, 0.5, "target", ha="center", va="center", rotation="vertical")
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()
