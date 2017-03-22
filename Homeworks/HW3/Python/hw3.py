from numpy import genfromtxt, diag, identity, zeros_like, ones_like, array
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

class SVMParams:
    w = matrix()
    b = 0
    xi = matrix()
    rho = 0
    lamb = 1

def cal_SVM(data, label, lamb, fig):
    train_data = matrix(data)
    train_label = matrix(label)
    Y = matrix(diag(label))
    I = matrix(identity(len(data)))

    Q = Y*train_data*train_data.trans()*Y + train_label*train_label.trans() + I/lamb
    p = matrix(zeros_like(train_label))
    G = -1*I
    h = matrix(zeros_like(train_label))
    A = matrix(ones_like(train_label)).trans()
    b = matrix(1.0)

    params = SVMParams()
    sol = solvers.qp(Q,p,G,h,A,b)
    params.alpha = sol['x']
    params.w = train_data.trans()*Y*params.alpha
    params.b = (params.alpha.trans()*train_label)[0]
    params.xi = params.alpha/lamb
    params.rho = min(Y*train_data*params.w+train_label*params.b+params.xi)
    params.lamb = lamb

    fig.scatter(train_data[:, 0], train_data[:, 1], c=train_label, cmap=plt.cm.coolwarm)

    minx = min(train_data[:, 0])
    maxx = max(train_data[:, 0])

    miny = (-params.w[0] * minx - params.b) / params.w[1]
    maxy = (-params.w[0] * maxx - params.b) / params.w[1]
    fig.plot([minx, maxx], [miny, maxy], color='green', label='Boundary')

    miny = (params.rho - params.w[0] * minx - params.b) / params.w[1]
    maxy = (params.rho - params.w[0] * maxx - params.b) / params.w[1]
    fig.plot([minx, maxx], [miny, maxy], color='red', label='Pos Margin')

    miny = (-params.rho - params.w[0] * minx - params.b) / params.w[1]
    maxy = (-params.rho - params.w[0] * maxx - params.b) / params.w[1]
    fig.plot([minx, maxx], [miny, maxy], color='blue', label='Neg Margin')

    fig.set_title("Lambda = "+str(lamb))
    fig.legend()

    return params

def test_SVM(data, label, params):
    test_data = matrix(data)
    test_label = matrix(label)
    Y = matrix(diag(label))
    preds = Y*test_data*params.w+test_label*params.b
    error = sum(pred < 0 for pred in preds)
    print('error with Lambda = '+str(params.lamb)+' : '+str(float(error) / len(label)))

if __name__ == '__main__':
    train_data = genfromtxt('./programming/train_data.csv', delimiter=' ')
    train_label = genfromtxt('./programming/train_label.csv')
    test_data = genfromtxt('./programming/test_data.csv', delimiter=' ')
    test_label = genfromtxt('./programming/test_label.csv')

    params1 = cal_SVM(train_data, train_label, 1, plt.subplot(2, 2, 1))
    params10 = cal_SVM(train_data, train_label, 10, plt.subplot(2, 2, 2))
    params100 = cal_SVM(train_data, train_label, 100, plt.subplot(2, 2, 3))
    params1000 = cal_SVM(train_data, train_label, 1000, plt.subplot(2, 2, 4))

    test_SVM(test_data, test_label, params1)
    test_SVM(test_data, test_label, params10)
    test_SVM(test_data, test_label, params100)
    test_SVM(test_data, test_label, params1000)

    plt.show()