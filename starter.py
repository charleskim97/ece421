import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import random
import math
import time

tf.disable_v2_behavior()


def loadData():
    with np.load('notMNIST.npz') as data:
        Data, Target = data['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx] / 255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target == posClass] = 1
        Target[Target == negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


def MSE(W, b, x, y, reg):
    N = x.shape[0]
    error = sum(np.square(np.matmul(x, W) + b - y)) / N + (reg / 2) * sum(np.square(W))
    return error


def gradMSE(W, b, x, y, reg):
    N = x.shape[0]
    z = np.matmul(x, W) + b
    c = 2 * (z - y)

    grad_w = (np.matmul(np.transpose(c), x) + 2 * reg * np.sum(np.square(W))) / N
    grad_b = sum(c) / N

    return np.transpose(grad_w), grad_b


def crossEntropyLoss(W, b, x, y, reg):
    N = x.shape[0]
    sig = 1 / (1 + np.exp(-1 * (np.matmul(x, W) + b)))
    loss = (np.sum(- (y * np.log(sig)) - (1 - y) * np.log(1 - sig))) / N + (reg / 2) * sum(np.square(W))
    return loss


def gradCE(W, b, x, y, reg):
    N = x.shape[0]
    sig = 1.0 / (1.0 + np.exp(-(np.matmul(x, W) + b)))
    return np.matmul(np.transpose(x), (sig - y)) / N + 2 * reg * W, np.sum(sig - y) / N


# Accuracy Helper function
def calc_Acc(W, b, x, y):
    results = []
    correct = 0

    pred = np.matmul(x, W) + b

    for i in pred:
        if i >= 0.5:
            results.append(1)
        else:
            results.append(0)

    res = results - y.squeeze()

    for i in range(len(res)):

        if res[i] == 0:
            correct += 1
    return correct / len(results)


def grad_descent(W, b, x, val_data, test_data, y, val_target, test_target, alpha, epochs, reg, error_tol, type):
    # Your implementation here
    train_losses = []
    val_losses = []
    test_losses = []

    train_accs = []
    val_accs = []
    test_accs = []

    weights = W

    if type == "MSE":
        for epoch in range(epochs):

            grad_w, grad_b = gradMSE(weights, b, x, y, reg)
            error = MSE(weights, b, x, y, reg)

            if error < error_tol:
                break

            weights = weights - grad_w * alpha
            b = b - grad_b * alpha

            train_loss = MSE(weights, b, x, y, reg)
            train_losses.append(train_loss)

            val_loss = MSE(weights, b, val_data, val_target, reg)
            val_losses.append(val_loss)

            test_loss = MSE(weights, b, test_data, test_target, reg)
            test_losses.append(test_loss)

            train_acc = calc_Acc(weights, b, x, y)
            val_acc = calc_Acc(weights, b, val_data, val_target)
            test_acc = calc_Acc(weights, b, test_data, test_target)

            train_accs.append(train_acc)
            val_accs.append(val_acc)
            test_accs.append(test_acc)

            print("Epoch: " + str(epoch))
            print("Training Accuracy: " + str(train_acc) + ", Validation Accuracy: " + str(
                val_acc) + ", Test Accuracy: " + str(test_acc))
            print("Training Loss: " + str(train_loss) + ", Validation Loss: " + str(val_loss) + ", Test Loss: " + str(
                test_loss))
    else:
        for epoch in range(epochs):
            grad_w, grad_b = gradCE(weights, b, x, y, reg)
            weights = weights - grad_w * alpha
            b = b - grad_b * alpha

            train_loss = crossEntropyLoss(weights, b, x, y, reg)
            train_losses.append(train_loss)

            if train_loss < error_tol:
                break

            val_loss = crossEntropyLoss(weights, b, val_data, val_target, reg)
            val_losses.append(val_loss)

            test_loss = crossEntropyLoss(weights, b, test_data, test_target, reg)
            test_losses.append(test_loss)

            train_acc = calc_Acc(weights, b, x, y)
            val_acc = calc_Acc(weights, b, val_data, val_target)
            test_acc = calc_Acc(weights, b, test_data, test_target)

            train_accs.append(train_acc)
            val_accs.append(val_acc)
            test_accs.append(test_acc)

            print("Epoch: " + str(epoch))
            print("Training Accuracy: " + str(train_acc) + ", Validation Accuracy: " + str(
                val_acc) + ", Test Accuracy: " + str(test_acc))
            print("Training Loss: " + str(train_loss) + ", Validation Loss: " + str(val_loss) + ", Test Loss: " + str(
                test_loss))

    if type == "MSE":
        print("MSE DONE")

    return weights, b, train_losses, train_accs, val_losses, val_accs, test_losses, test_accs


def leastSquares(x, y):
    xTx = np.matmul(np.transpose(x), x)
    xTxinv = np.linalg.inv(xTx)
    W = np.matmul(xTxinv, np.matmul(np.transpose(x), y))
    return W


def buildGraph(loss, b1, b2, e):
    tf.set_random_seed(421)

    W = tf.Variable(tf.truncated_normal(shape=(784, 1), mean=0, stddev=0.5, dtype=tf.float64, name="weights"))
    b = tf.Variable(1, dtype=tf.float64, name='bias')

    x = tf.placeholder(tf.float64, shape=(None, 784), name='x')
    y = tf.placeholder(tf.float64, shape=(None, 1), name='y')

    reg = tf.placeholder(tf.float64, name='lam')
    alpha = tf.placeholder(tf.float64, name='lr')

    pred = tf.matmul(x, W) + b

    regularizer = tf.nn.l2_loss(W)

    if loss == "MSE":

        losses = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=pred))
        losses = tf.cast(losses, tf.float64)
        l = losses + (reg / 2) * regularizer




    elif loss == "CE":

        losses = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=pred))
        l = losses + (reg / 2) * regularizer

    optimizer = tf.train.AdamOptimizer(learning_rate=alpha, beta1=b1, beta2=b2, epsilon=e).minimize(l)

    return W, b, x, y, pred, l, optimizer, reg, alpha


def sgd(batchSize, epochs, a, r, lossType, b1, b2, e):
    train_data, val_data, test_data, train_target, val_target, test_target = loadData()

    train_data = train_data.reshape((train_data.shape[0], 784))
    val_data = val_data.reshape(val_data.shape[0], 784)
    test_data = test_data.reshape(test_data.shape[0], 784)

    W, b, x, y, pred, l, optimizer, reg, alpha = buildGraph(lossType, b1, b2, e)
    init_op = tf.global_variables_initializer()

    n = train_data.shape[0]
    num_batches = int(n / batchSize)

    train_losses = []
    val_losses = []
    test_losses = []

    train_accs = []
    val_accs = []
    test_accs = []
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(epochs):
            # Shuffle every epoch
            # np.random.seed(i)
            index = np.random.permutation(len(train_data))
            x_shuffled = train_data[index]
            y_shuffled = train_target[index]

            for k in range(num_batches):
                x_batch = x_shuffled[k * num_batches:k * num_batches + batchSize, :]
                y_batch = y_shuffled[k * num_batches:k * num_batches + batchSize, :]

                optim, w1, bias1, l1 = sess.run([optimizer, W, b, l],
                                                feed_dict={x: x_batch, y: y_batch, reg: r, alpha: a})
            train_losses.append(l1)
            train_acc = (calc_Acc(w1, bias1, train_data, train_target))
            train_accs.append(train_acc)

            if lossType == "MSE":
                v_loss = MSE(w1, bias1, val_data, val_target, r)
                val_losses.append(v_loss)
                val_acc = calc_Acc(w1, bias1, val_data, val_target)
                val_accs.append(val_acc)

                t_loss = MSE(w1, bias1, test_data, test_target, r)
                test_losses.append(t_loss)
                test_acc = calc_Acc(w1, bias1, test_data, test_target)
                test_accs.append(test_acc)

            elif lossType == "CE":
                v_loss = crossEntropyLoss(w1, bias1, val_data, val_target, r)
                val_losses.append(v_loss)
                val_acc = calc_Acc(w1, bias1, val_data, val_target)
                val_accs.append(val_acc)

                t_loss = crossEntropyLoss(w1, bias1, test_data, test_target, r)
                test_losses.append(t_loss)
                test_acc = calc_Acc(w1, bias1, test_data, test_target)
                test_accs.append(test_acc)

            print("Epoch: " + str(i))
            print("Training Accuracy: " + str(train_acc) + ", Validation Accuracy: " + str(
                val_acc) + ", Test Accuracy: " + str(test_acc))
            print("Training Loss: " + str(l1) + ", Validation Loss: " + str(v_loss) + ", Test Loss: " + str(
                t_loss))

    return W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs


def plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs, t1, t2):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(t1)
    plt.legend()
    plt.show()

    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(t2)
    plt.legend()
    plt.show()

# 1 Linear Regression
# Part 1.3 Tuning the Learning Rate
train_data, val_data, test_data, train_target, val_target, test_target = loadData()
random.seed(421)
lossType = "MSE"
train_data = train_data.reshape((train_data.shape[0], 784))
val_data = val_data.reshape(val_data.shape[0], 784)
test_data = test_data.reshape(test_data.shape[0], 784)
W = np.random.normal(0, 0.5, (784, 1))
b = np.zeros(1)
error_tol = 10 ^ (-7)
epochs = 5000
alpha = 0.005
reg = 0

b1 = 0.9
b2 = 0.999
e = math.exp(-8)

'''
# Part 1
# 5000 epochs, lambda = 0, a = 0.005
t1 = time.time()
alpha = 0.005
reg = 0
trained_weights, trained_bias, train_losses, train_accs, val_losses, val_accs, test_losses, test_accs = grad_descent(W,
                                                                                                                     b,
                                                                                                                     train_data,
                                                                                                                     val_data,
                                                                                                                     test_data,
                                                                                                                     train_target,
                                                                                                                     val_target,
                                                                                                                     test_target,
                                                                                                                     alpha,
                                                                                                                     epochs,
                                                                                                                     reg,
                                                                                                                     error_tol, lossType)
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "MSE: Loss vs Epochs (lr = 0.005, reg = 0)", "MSE: Acc vs Epochs (lr = 0.005, reg = 0)")
print("Runtime")
print(time.time() - t1)
# a = 0.001
'''

'''
alpha = 0.001
trained_weights, trained_bias, train_losses, train_accs, val_losses, val_accs, test_losses, test_accs = grad_descent(W,
                                                                                                                     b,
                                                                                                                     train_data,
                                                                                                                     val_data,
                                                                                                                     test_data,
                                                                                                                     train_target,
                                                                                                                     val_target,
                                                                                                                     test_target,
                                                                                                                     alpha,
                                                                                                                     epochs,
                                                                                                                     reg,
                                                                                                                     error_tol, lossType)
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "MSE: Loss vs Epochs (lr = 0.001, reg = 0)", "MSE: Acc vs Epochs (lr = 0.001, reg = 0)")

# a = 0.0001
alpha = 0.0001
trained_weights, trained_bias, train_losses, train_accs, val_losses, val_accs, test_losses, test_accs = grad_descent(W,
                                                                                                                     b,
                                                                                                                     train_data,
                                                                                                                     val_data,
                                                                                                                     test_data,
                                                                                                                     train_target,
                                                                                                                     val_target,
                                                                                                                     test_target,
                                                                                                                     alpha,
                                                                                                                     epochs,
                                                                                                                     reg,
                                                                                                                     error_tol, lossType)
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "MSE: Loss vs Epochs (lr = 0.0001, reg = 0)", "MSE: Acc vs Epochs (lr = 0.0001, reg = 0)")
'''
# Part 1.4 Generalization
# lambda = 0.001, alpha = 0.005
'''
lossType = "MSE"
alpha = 0.005
reg = 0.001
trained_weights, trained_bias, train_losses, train_accs, val_losses, val_accs, test_losses, test_accs = grad_descent(W,
                                                                                                                     b,
                                                                                                                     train_data,
                                                                                                                     val_data,
                                                                                                                     test_data,
                                                                                                                     train_target,
                                                                                                                     val_target,
                                                                                                                     test_target,
                                                                                                                     alpha,
                                                                                                                     epochs,
                                                                                                                     reg,
                                                                                                                     error_tol, lossType)
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "MSE: Loss vs Epochs (lr = 0.005, reg = 0.001)", "MSE: Acc vs Epochs (lr = 0.005, reg = 0.001)")
'''

'''
# lambda = 0.1
reg = 0.1
trained_weights, trained_bias, train_losses, train_accs, val_losses, val_accs, test_losses, test_accs = grad_descent(W,
                                                                                                                     b,
                                                                                                                     train_data,
                                                                                                                     val_data,
                                                                                                                     test_data,
                                                                                                                     train_target,
                                                                                                                     val_target,
                                                                                                                     test_target,
                                                                                                                     alpha,
                                                                                                                     epochs,
                                                                                                                     reg,
                                                                                                                     error_tol, lossType)
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "MSE: Loss vs Epochs (lr = 0.005, reg = 0.1)", "MSE: Acc vs Epochs (lr = 0.005, reg = 0.1)")
'''
'''
# lambda = 0.5
reg = 0.5
trained_weights, trained_bias, train_losses, train_accs, val_losses, val_accs, test_losses, test_accs = grad_descent(W,
                                                                                                                     b,
                                                                                                                     train_data,
                                                                                                                     val_data,
                                                                                                                     test_data,
                                                                                                                     train_target,
                                                                                                                     val_target,
                                                                                                                     test_target,
                                                                                                                     alpha,
                                                                                                                     epochs,
                                                                                                                     reg,
                                                                                                                     error_tol,
                                                                                                                     lossType)
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "MSE: Loss vs Epochs (lr = 0.005, reg = 0.5)", "MSE: Acc vs Epochs (lr = 0.005, reg = 0.5)")
'''

'''
# Part 1.5 Comparing Batch GD with normal equation
# Least Squares
t1 = time.time()
w_least_squares = leastSquares(train_data, train_target)
print("Least Squares Train acc:")
print(calc_Acc(w_least_squares, b, train_data, train_target))
print("Least Squares Val acc:")
print(calc_Acc(w_least_squares, b, val_data, val_target))
print("Least Squares Test acc:")
print(calc_Acc(w_least_squares, b, test_data, test_target))

print("MSE Train Loss: ")
print(MSE(w_least_squares, 0, train_data, train_target, 0))

print("MSE Train Loss: ")
print(MSE(w_least_squares, 0, val_data, val_target, 0))

print("MSE Train Loss: ")
print(MSE(w_least_squares, 0, test_data, test_target, 0))
print("Least Squares Time: " + str(time.time() - t1))
'''

# 2 Logistic Regression

# Part 2.2 Learning
# lambda = 0.1, alpha = 0.005
'''
lossType = "CE"
reg = 0.1
alpha = 0.005
epochs = 5000
trained_weights, trained_bias, train_losses, train_accs, val_losses, val_accs, test_losses, test_accs = grad_descent(W,
                                                                                                                     b,
                                                                                                                     train_data,
                                                                                                                     val_data,
                                                                                                                     test_data,
                                                                                                                     train_target,
                                                                                                                     val_target,
                                                                                                                     test_target,
                                                                                                                     alpha,
                                                                                                                     epochs,
                                                                                                                     reg,
                                                                                                                     error_tol,
                                                                                                                     lossType)
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "CE: Loss vs Epochs (lr = 0.005, reg = 0.1)",
      "CE: Accuracy vs Epochs (lr = 0.005, reg = 0.1)")

'''
'''
# Part 2.3 Comparison to Linear Regression
# lambda = 0, alpha = 0.005
reg = 0
alpha = 0.005
lossType = "CE"
trained_weights, trained_bias, train_losses_CE, train_accs, val_losses, val_accs, test_losses, test_accs = grad_descent(
    W,
    b,
    train_data,
    val_data,
    test_data,
    train_target,
    val_target,
    test_target,
    alpha,
    epochs,
    reg,
    error_tol, lossType)
lossType = "MSE"
trained_weights, trained_bias, train_losses_MSE, train_accs, val_losses, val_accs, test_losses, test_accs = grad_descent(
    W,
    b,
    train_data,
    val_data,
    test_data,
    train_target,
    val_target,
    test_target,
    alpha,
    epochs,
    reg,
    error_tol, lossType)

plt.plot(train_losses_MSE, label='MSE Loss')
plt.plot(train_losses_CE, label='CE Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("MSE vs CE: Loss vs Epochs (lr = 0.005, reg = 0")
plt.legend()
plt.show()
'''

'''
# Plot code for part 3
# Part 3.2 Implementing Stochastic Gradient Descent
# MSE, Batch size = 500, Epochs = 700,  alpha = 0.001


b1 = 0.9
b2 = 0.999
e = math.exp(-8)


W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=500, epochs=700, a=0.001,
                                                                                   r=0.0, lossType='MSE', b1 = b1, b2 =b2, e =e )
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "SGD: Loss vs Epochs (batch size = 500, lr = 0.001, reg = 0)",
      "SGD: Accuracy vs Epochs (batch size = 500, lr = 0.001, reg = 0)")

'''
'''
# Part 3.3 Batch Size Investigation
# Batch size = 100
W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=100, epochs=700, a=0.001,
                                                                                   r=0.0, lossType='MSE', b1 = b1, b2 =b2, e =e )
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "SGD: Loss vs Epochs (batch size = 100, lr = 0.001, reg = 0)",
      "SGD: Accuracy vs Epochs (batch size = 100, lr = 0.001, reg = 0)")
# Batch size = 700
W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=700, epochs=700, a=0.001,
                                                                                   r=0.0, lossType='MSE', b1 = b1, b2 =b2, e =e )
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "SGD: Loss vs Epochs (batch size = 700, lr = 0.001, reg = 0)",
      "SGD: Accuracy vs Epochs (batch size = 700, lr = 0.001, reg = 0)")


# Batch size = 1750
W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=1750, epochs=700, a=0.001,
                                                                                   r=0.0, lossType='MSE', b1 = b1, b2 =b2, e =e )
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "SGD: Loss vs Epochs (batch size = 1750, lr = 0.001, reg = 0)",
      "SGD: Accuracy vs Epochs (batch size = 1750, lr = 0.001, reg = 0)")
'''
'''
# Part 4 Hyperparameter Investigation
# Defaults b1 = 0.9, b2 = 0.999, e = math.exp(-8)
# b1 = 0.95
b1 = 0.95
b2 = 0.999
e = math.exp(-8)
W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=500, epochs=700, a=0.001,
                                                                                   r=0.0, lossType='MSE', b1=b1, b2=b2,
                                                                                   e=e)
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "SGD: Loss vs Epochs (beta1 = 0.95)",
      "SGD: Accuracy vs Epochs (beta1 = 0.95)")

# b1 = 0.99
b1 = 0.99
b2 = 0.999
e = math.exp(-8)
W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=500, epochs=700, a=0.001,
                                                                                   r=0.0, lossType='MSE', b1=b1, b2=b2,
                                                                                   e=e)
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "SGD: Loss vs Epochs (beta1 = 0.99)",
      "SGD: Accuracy vs Epochs (beta1 = 0.99)")

# b2 = 0.99
b1 = 0.9
b2 = 0.99
e = math.exp(-8)
W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=500, epochs=700, a=0.001,
                                                                                   r=0.0, lossType='MSE', b1=b1, b2=b2,
                                                                                   e=e)
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "SGD: Loss vs Epochs (beta2 = 0.99)",
      "SGD: Accuracy vs Epochs (beta2 = 0.99)")

# b2 = 0.9999
b1 = 0.9
b2 = 0.9999
e = math.exp(-8)
W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=500, epochs=700, a=0.001,
                                                                                   r=0.0, lossType='MSE', b1=b1, b2=b2,
                                                                                   e=e)
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "SGD: Loss vs Epochs (beta2 = 0.9999)",
      "SGD: Accuracy vs Epochs (beta2 = 0.9999)")

# e = 1e-09
b1 = 0.9
b2 = 0.999
e = math.exp(-9)
W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=500, epochs=700, a=0.001,
                                                                                   r=0.0, lossType='MSE', b1=b1, b2=b2,
                                                                                   e=e)
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "SGD: Loss vs Epochs (e = e^(-9))",
      "SGD: Accuracy vs Epochs (e = e^(-9))")


# e = 1e-4
b1 = 0.9
b2 = 0.999
e = math.exp(-4)
W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=500, epochs=700, a=0.001,
                                                                                   r=0.0, lossType='MSE', b1=b1, b2=b2,
                                                                                   e=e)
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "SGD: Loss vs Epochs (e = e^(-4))",
      "SGD: Accuracy vs Epochs (e = e^(-4))")
'''
# 3.5 Cross Entropy Loss Investigation
# 3.2 repeat
# Part 3.2 Implementing Stochastic Gradient Descent
# MSE, Batch size = 500, Epochs = 700,  alpha = 0.001
'''
b1 = 0.9
b2 = 0.999
e = math.exp(-8)


W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=500, epochs=700, a=0.001,
                                                                                   r=0.0, lossType='CE', b1 = b1, b2 =b2, e =e )
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "SGD: Loss vs Epochs (batch size = 500, lr = 0.001, reg = 0)",
      "SGD: Accuracy vs Epochs (batch size = 500, lr = 0.001, reg = 0)")



# 3.3 repeat
# Part 3.3 Batch Size Investigation
# Batch size = 100

b1 = 0.9
b2 = 0.999
e = math.exp(-8)


W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=100, epochs=700, a=0.001,
                                                                                   r=0.0, lossType='CE', b1 = b1, b2 =b2, e =e )
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "SGD CE: Loss vs Epochs (batch size = 100, lr = 0.001, reg = 0)",
      "SGD CE: Accuracy vs Epochs (batch size = 100, lr = 0.001, reg = 0)")
# Batch size = 700
W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=700, epochs=700, a=0.001,
                                                                                   r=0.0, lossType='CE', b1 = b1, b2 =b2, e =e )
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "SGD CE: Loss vs Epochs (batch size = 700, lr = 0.001, reg = 0)",
      "SGD CE: Accuracy vs Epochs (batch size = 700, lr = 0.001, reg = 0)")
      

# Batch size = 1750
W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=1750, epochs=700, a=0.001,
                                                                                   r=0.0, lossType='CE', b1 = b1, b2 =b2, e =e )
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "SGD CE: Loss vs Epochs (batch size = 1750, lr = 0.001, reg = 0)",
      "SGD CE: Accuracy vs Epochs (batch size = 1750, lr = 0.001, reg = 0)")


# Part 4 repeat
# Part 4 Hyperparameter Investigation
# Defaults b1 = 0.9, b2 = 0.999, e = math.exp(-8)
# b1 = 0.95
b1 = 0.95
b2 = 0.999
e = math.exp(-8)
W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=500, epochs=700, a=0.001,
                                                                                   r=0.0, lossType='CE', b1=b1, b2=b2,
                                                                                   e=e)
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "SGD CE: Loss vs Epochs (beta1 = 0.95)",
      "SGD CE: Accuracy vs Epochs (beta1 = 0.95)")

# b1 = 0.99
b1 = 0.99
b2 = 0.999
e = math.exp(-8)
W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=500, epochs=700, a=0.001,
                                                                                   r=0.0, lossType='CE', b1=b1, b2=b2,
                                                                                   e=e)
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "SGD CE: Loss vs Epochs (beta1 = 0.99)",
      "SGD CE: Accuracy vs Epochs (beta1 = 0.99)")

# b2 = 0.99
b1 = 0.9
b2 = 0.99
e = math.exp(-8)
W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=500, epochs=700, a=0.001,
                                                                                   r=0.0, lossType='CE', b1=b1, b2=b2,
                                                                                   e=e)
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "SGD CE: Loss vs Epochs (beta2 = 0.99)",
      "SGD CE: Accuracy vs Epochs (beta2 = 0.99)")

# b2 = 0.9999
b1 = 0.9
b2 = 0.9999
e = math.exp(-8)
W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=500, epochs=700, a=0.001,
                                                                                   r=0.0, lossType='CE', b1=b1, b2=b2,
                                                                                   e=e)
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "SGD CE: Loss vs Epochs (beta2 = 0.9999)",
      "SGD CE: Accuracy vs Epochs (beta2 = 0.9999)")

# e = 1e-09
b1 = 0.9
b2 = 0.999
e = math.exp(-9)
W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=500, epochs=700, a=0.001,
                                                                                   r=0.0, lossType='CE', b1=b1, b2=b2,
                                                                                   e=e)
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "SGD CE: Loss vs Epochs (e = e^(-9))",
      "SGD CE: Accuracy vs Epochs (e = e^(-9))")


# e = 1e-4
b1 = 0.9
b2 = 0.999
e = math.exp(-4)
W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=500, epochs=700, a=0.001,
                                                                                   r=0.0, lossType='CE', b1=b1, b2=b2,
                                                                                   e=e)
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
      "SGD CE: Loss vs Epochs (e = e^(-4))",
      "SGD CE: Accuracy vs Epochs (e = e^(-4))")
'''