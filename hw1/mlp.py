import argparse
import numpy as np
import matplotlib.pyplot as plt
import utils

# input and output dimensions
input_dim = 784
output_dim = 10

def ReLU(z):
    return np.maximum(0.0,z)

def softmax(z):
    z_temp = np.exp(z)
    return z_temp / np.sum(z_temp, axis=1, keepdims=True)


def calc_loss_and_grad(x, y, w1, b1, w2, b2, eval_only=False):
    """Forward Propagation and Backward Propagation.

    Given a mini-batch of images x, associated labels y, and a set of parameters, compute the
    cross-entropy loss and gradients corresponding to these parameters.

    :param x: images of one mini-batch.
    :param y: labels of one mini-batch.
    :param w1: weight parameters of layer 1.
    :param b1: bias parameters of layer 1.
    :param w2: weight parameters of layer 2.
    :param b2: bias parameters of layer 2.
    :param eval_only: if True, only return the loss and predictions of the MLP.
    :return: a tuple of (loss, db2, dw2, db1, dw1)
    """

    # TODO
    z1=x@w1+b1
    h1=ReLU(z1)
    z2=h1@w2+b2
    y_hat=softmax(z2)

    # forward pass
    loss = -np.sum(y * np.log(y_hat)) / len(y)

    if eval_only:
        return loss, y_hat

    # TODO
    # backward pass
    delta=(y_hat-y)/len(y)
    dw2=h1.T@delta
    db2=np.sum(delta, axis=0)
    dw1= x.T@(delta@(w2.T)*(z1>0.0))
    db1=np.sum(delta@(w2.T)*(z1>0.0), axis=0)
    # print(x.shape,y.shape,w1.shape,w2.shape,b1.shape,b2.shape,dw1.shape,dw2.shape,db1.shape,db2.shape)
    # print(delta.shape)
    return loss, db2, dw2, db1, dw1


def train(train_x, train_y, test_x, text_y, args: argparse.Namespace):
    """Train the network.

    :param train_x: images of the training set.
    :param train_y: labels of the training set.
    :param test_x: images of the test set.
    :param text_y: labels of the test set.
    :param args: a dict of hyper-parameters.
    """

    # TODO
    #  randomly initialize the parameters (weights and biases)
    #  Xaiver initialize
    hidden_dim=args.hidden_dim
    w1, b1, w2, b2 = np.random.normal(loc=0.0,scale=np.sqrt(2./(input_dim+hidden_dim)),size=[input_dim,hidden_dim]), \
                    np.random.normal(loc=0.0,scale=np.sqrt(2./(input_dim+hidden_dim)),size=[hidden_dim]), \
                    np.random.normal(loc=0.0,scale=np.sqrt(2./(output_dim+hidden_dim)),size=[hidden_dim,output_dim]),  \
                    np.random.normal(loc=0.0,scale=np.sqrt(2./(output_dim+hidden_dim)),size=[output_dim])

    print('Start training:')
    print_freq = 100
    loss_curve = []

    for epoch in range(args.epochs):
        # train for one epoch
        print("[Epoch #{}]".format(epoch))

        # random shuffle dataset
        dataset = np.hstack((train_x, train_y))
        np.random.shuffle(dataset)
        train_x = dataset[:, :input_dim]
        train_y = dataset[:, input_dim:]

        n_iterations = train_x.shape[0] // args.batch_size

        for i in range(n_iterations):
            # load a mini-batch
            x_batch = train_x[i * args.batch_size: (i + 1) * args.batch_size, :]
            y_batch = train_y[i * args.batch_size: (i + 1) * args.batch_size, :]

            # TODO
            # compute loss and gradient

            loss, db2, dw2, db1, dw1 = calc_loss_and_grad(x_batch,y_batch,w1,b1,w2,b2)

            # TODO
            # update parameters
            b2=b2-args.learning_rate*db2
            w2=w2-args.learning_rate*dw2
            b1=b1-args.learning_rate*db1
            w1=w1-args.learning_rate*dw1

            loss_curve.append(loss)
            if i % print_freq == 0:
                print('[Iteration #{}/{}] [Loss #{:4f}]'.format(i, n_iterations, loss))

    # show learning curve
    plt.title('Training Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(range(len(loss_curve)), loss_curve)
    plt.show()

    # evaluate on the training set
    loss, y_hat = calc_loss_and_grad(train_x, train_y, w1, b1, w2, b2, eval_only=True)
    predictions = np.argmax(y_hat, axis=1)
    labels = np.argmax(train_y, axis=1)
    accuracy = np.sum(predictions == labels) / train_x.shape[0]
    print('Top-1 accuracy on the training set', accuracy)

    # evaluate on the test set
    loss, y_hat = calc_loss_and_grad(test_x, text_y, w1, b1, w2, b2, eval_only=True)
    predictions = np.argmax(y_hat, axis=1)
    labels = np.argmax(text_y, axis=1)
    accuracy = np.sum(predictions == labels) / test_x.shape[0]
    print('Top-1 accuracy on the test set', accuracy)


def main(args: argparse.Namespace):
    # print hyper-parameters
    print('Hyper-parameters:')
    print(args)

    # load training set and test set
    train_x, train_y = utils.load_data("train")
    test_x, text_y = utils.load_data("test")
    print('Dataset information:')
    print("training set size: {}".format(len(train_x)))
    print("test set size: {}".format(len(test_x)))

    # check your implementation of backward propagation before starting training
    # utils.check_grad(calc_loss_and_grad)

    # train the network and report the accuracy on the training and the test set
    train(train_x, train_y, test_x, text_y, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multilayer Perceptron')
    parser.add_argument('--hidden-dim', default=128, type=int,
                        help='hidden dimension of the Multilayer Perceptron')
    parser.add_argument('--learning_rate', default=0.0005, type=float,
                        help='learning rate')
    parser.add_argument('--batch-size', default=100, type=int,
                        help='mini-batch size')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    args = parser.parse_args()
    main(args)
    # x=np.ones(shape=[5,3])
    # w1=np.ones()
