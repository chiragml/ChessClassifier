#!/bin/env python3
import pickle
import sys
import random
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score


def repl():
    import code
    code.InteractiveConsole(locals=globals()).interact()


if __name__ == "__main__":
    n = len(sys.argv)
    if n == 1:
        print('Usage: ', sys.argv[0], ' <infile> <outfile>')
    infile, outfile = sys.argv[1:]
    data = pickle.load(open(infile, 'rb'))
    XY = list(zip(data['loss'] + data['win'],
                  [0]*len(data['loss']) + [1]*len(data['win'])))
    # XY += list(zip(data['draw']+data['draw'],
    #                [0]*len(data['draw'])+[1]*len(data['draw'])))
    random.shuffle(XY)
    n = int(len(XY)*0.5)
    train = list(zip(*XY[:n]))
    test = list(zip(*XY[n:]))
    print('training:')
    model = lr(max_iter=1000).fit(train[0], train[1])
    pickle.dump(model, open(outfile, 'wb'))
    print('model saved to ', outfile)
    print('train ac', accuracy_score(train[1], model.predict(train[0])))
    print('test ac ', accuracy_score(test[1], model.predict(test[0])))
