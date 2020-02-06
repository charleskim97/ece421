import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import random

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
    # Your implementation here
    # Reshape x matrix from 3D to 2D
    N = x.shape[0]
    error = sum(np.square(np.matmul(x, W) + b - y)) / N + reg / 2 * sum(np.square(W))
    return error


def gradMSE(W, b, x, y, reg):
    # Your implementation here

    N = x.shape[0]

    z = np.matmul(x, W) + b
    c = 2 * (z - y)

    grad_w = (np.matmul(np.transpose(c), x) + 2 * reg * np.sum(np.matmul(W, np.transpose(W)))) / N
    grad_b = sum((2 * (z - y))) / N

    return np.transpose(grad_w), grad_b


def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here

    N = x.shape[0]

    # Calculate sigmoid function
    sig = 1 / (1 + np.exp(-1 * (np.matmul(x, W) + b)))

    # Calculate loss
    loss = (np.sum(- (y * np.log(sig)) - (1 - y) * np.log(1 - sig))) / N + (reg / 2) * sum(np.square(W))
    return loss


def gradCE(W, b, x, y, reg):
    # Your implementation here

    N = x.shape[0]
    sig = 1.0 / (1.0 + np.exp(-(np.matmul(x, W) + b)))

    return np.matmul(np.transpose(x), (sig - y)) / N + 2 * reg * W, np.sum(sig - y) / N


# Accuracy Helper function
def calc_Acc(W, x, y):
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


def grad_descent(W, b, x, val_data, test_data, y, val_target, test_target, alpha, epochs, reg, error_tol):
    # Your implementation here
    train_losses = []
    val_losses = []
    test_losses = []

    train_accs = []
    val_accs = []
    test_accs = []

    weights = W
    type = "MSE"
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

            train_acc = calc_Acc(weights, x, y)
            val_acc = calc_Acc(weights, val_data, val_target)
            test_acc = calc_Acc(weights, test_data, test_target)

            train_accs.append(train_acc)
            val_accs.append(val_acc)
            test_accs.append(test_acc)

            print("Epoch: " + str(epoch))
            print("Training Accuracy: " + str(train_acc) + ", Validation Accuracy: " + str(
                val_acc) + ", Test Accuracy: " + str(test_acc))
            print("Training Loss: " + str(train_loss) + ", Validation Loss: " + str(val_loss) + ", Test Loss: " + str(
                test_loss))
    else:
        print("CE")
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

            train_acc = calc_Acc(weights, x, y)
            val_acc = calc_Acc(weights, val_data, val_target)
            test_acc = calc_Acc(weights, test_data, test_target)

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
    print(x.shape)
    print(y.shape)
    xTx = np.matmul(np.transpose(x), x)
    xTxinv = np.linalg.inv(xTx)
    W = np.matmul(xTxinv, np.matmul(np.transpose(x), y))
    return W


def buildGraph(loss, val_data, val_target, test_data, test_target):
    # Initialize weight and bias tensors
    tf.set_random_seed(421)

    W = tf.Variable(tf.random.truncated_normal(shape=(784, 1), mean=0, stddev=0.5, dtype=tf.float64, name="weights"))
    b = tf.Variable(tf.random.truncated_normal(shape=(1,), mean=0, stddev=0.5, dtype=tf.float64, name="bias"))

    x = tf.placeholder(tf.float64, shape=(None, 784), name='x')
    y = tf.placeholder(tf.float64, shape=(None, 1), name='y')

    '''
    val_data = tf.placeholder(tf.float32, shape=(None, 784), name='v_data')
    val_target = tf.placeholder(tf.float32, shape=(None, 1), name='v_target')
    test_data = tf.placeholder(tf.float32, shape=(None, 784), name='t_data')
    test_target = tf.placeholder(tf.float32, shape=(None, 1), name='t_target')
    '''

    reg = tf.placeholder(tf.float64, name='lam')
    alpha = tf.placeholder(tf.float64, name='lr')

    pred = tf.matmul(x, W) + b

    val_pred = tf.matmul(val_data, W) + b
    test_pred = tf.matmul(test_data, W) + b

    regularizer = tf.nn.l2_loss(W)


    if loss == "MSE":
        losses = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=pred))
        losses = tf.cast(losses, tf.float64)
        l = tf.reduce_mean(losses + (reg / 2) * regularizer)

        val_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=val_target, predictions=val_pred))
        val_loss = tf.cast(val_loss, tf.float64)
        vl = tf.reduce_mean(val_loss + (reg / 2) * regularizer)

        test_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=test_target, predictions=test_pred))
        test_loss = tf.cast(test_loss, tf.float64)
        tl = tf.reduce_mean(test_loss + (reg / 2) * regularizer)

    elif loss == "CE":

        losses = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=pred))
        l = tf.reduce_mean(losses + (reg / 2) * regularizer)
        val_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=val_target, logits=val_pred))
        vl = tf.reduce_mean(val_loss + (reg / 2) * regularizer)
        test_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=test_target, logits=test_pred))
        tl = tf.reduce_mean(test_loss + (reg / 2) * regularizer)

    optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(l)

    return W, b, x, y, pred, l, optimizer, reg, alpha, vl, tl


def sgd(batchSize, epochs, a, r, lossType):

    train_data, v_data, t_data, t_target, v_target, t_target = loadData()

    tf.cast(v_data, dtype=tf.float64)
    tf.cast(t_data, dtype=tf.float64)
    tf.cast(val_target, dtype=tf.float64)
    tf.cast(test_target, dtype=tf.float64)
    train_data = train_data.reshape((train_data.shape[0], 784))
    v_data = val_data.reshape(v_data.shape[0], 784)
    t_data = test_data.reshape(t_data.shape[0], 784)
    W, b, x, y, pred, l, optimizer, reg, alpha, vl, tl = buildGraph(lossType, v_data, val_target, t_data,
                                                                    test_target)
    init_op = tf.global_variables_initializer()

    n = train_data.shape[0]
    num_batches = int(n / batchSize)

    train_losses = []
    val_losses = []
    test_losses = []

    train_accs = []
    val_accs = []
    test_accs = []

    for i in range(epochs):
        # Shuffle every epoch
        np.random.seed(i)
        index = np.random.permutation(len(train_data))
        x_shuffled = train_data[index]
        y_shuffled = train_target[index]

        for k in range(num_batches):
            x_batch = x_shuffled[i * num_batches:i * num_batches + batchSize, :]
            y_batch = y_shuffled[i * num_batches:i * num_batches + batchSize, :]

            with tf.Session() as sess:
                sess.run(init_op)

                opt1, w1, bias1, l1, vl1, tl1 = sess.run([optimizer, W, b, l, vl, tl],
                                                  feed_dict={x: x_batch, y: y_batch, reg: r, alpha: a})


        train_losses.append(l1)
        #train_losses.append(crossEntropyLoss(w1, bias1, train_data, train_target, r))
        train_acc = (calc_Acc(w1, train_data, train_target))
        train_accs.append(train_acc)

        #val_losses.append(vl1)
        v_loss = crossEntropyLoss(w1, bias1, v_data, val_target, r)
        val_losses.append(v_loss)
        val_acc = calc_Acc(w1, v_data, val_target)
        val_accs.append(val_acc)

        #test_losses.append(tl1)
        t_loss = crossEntropyLoss(w1, bias1, t_data, test_target, r)
        test_losses.append(t_loss)
        test_acc = calc_Acc(w1, t_data, test_target)
        test_accs.append(test_acc)

        with tf.Session():
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


train_data, val_data, test_data, train_target, val_target, test_target = loadData()
random.seed(421)
losstype = "CE"
train_data = train_data.reshape((train_data.shape[0], 784))
val_data = val_data.reshape(val_data.shape[0], 784)
test_data = test_data.reshape(test_data.shape[0], 784)

W = np.random.normal(0, 0.5, (784, 1))
b = np.zeros(1)
epochs = 4
reg = 0.5
error_tol = 10 ^ (-7)
alpha = 0.005

'''
# sgd(batchSize=500, epochs=10, a = 0.001, r = 0.1, lossType = 'MSE')
# trained_weights, trained_bias, train_losses, train_accs, val_losses, val_accs, test_losses, test_accs = grad_descent(W,
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
                                                                                                                     error_tol)
                                                                                        '''

# Plot code for part 1
# plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs, "Title 1", "Title 2")
# Least Squares
# w_least_squares = leastSquares(train_data, train_target)
# print(calc_Acc(w_least_squares, train_data, train_target))

# Plot code for part 3
W, b, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = sgd(batchSize=500, epochs=10, a=0.001, r=0.0, lossType='MSE')
plots(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs, "Loss vs Epochs", "Accuracy vs Epochs")