#!/usr/bin/env/python

import numpy as np
import tensorflow as tf
import queue
import threading

SMALL_NUMBER = 1e-7


def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)


class ThreadedIterator:
    """An iterator object that computes its elements in a parallel thread to be ready to be consumed.
    The iterator should *not* return None"""

    def __init__(self, original_iterator, max_queue_size: int=2):
        self.__queue = queue.Queue(maxsize=max_queue_size)
        self.__thread = threading.Thread(target=lambda: self.worker(original_iterator))
        self.__thread.start()

    def worker(self, original_iterator):
        for element in original_iterator:
            assert element is not None, 'By convention, iterator elements much not be None'
            self.__queue.put(element, block=True)
        self.__queue.put(None, block=True)

    def __iter__(self):
        next_element = self.__queue.get(block=True)
        while next_element is not None:
            yield next_element
            next_element = self.__queue.get(block=True)
        self.__thread.join()

def pickResBaseOnFile(graphPred, y, graphFileHash):
    res = {}
    for index in range(len(graphFileHash)):
        fileHash = graphFileHash[index]
        if res.get(fileHash) == None:
            res[fileHash] = [[], []]
        res[fileHash][0].append(graphPred[index])
        res[fileHash][1].append(y[index])
    return res

def returnTopNRes(preds, indices, n, ifkeep=False):
    pos = 0
    res = []
    for i in range(len(indices)):
        labelInd = indices[i]
        predRes = []
        for predIndex in range(pos, pos+labelInd):
            predRes.append(preds[predIndex])
        pos += labelInd
        sortedIndex = sorted(range(len(predRes)), key=lambda k:predRes[k], reverse = True)
        predRes = sortedIndex[:n]
        if ifkeep:
            resArr = [0 for j in range(labelInd)]
            for j in predRes:
                resArr[j] = 1
            predRes = resArr
        res.append(predRes)
    return res

def returnSEMet(res):
    fgp = res[0]
    fy = res[1]
    fgp = np.array(fgp)
    fy = np.array(fy)
    TP = np.count_nonzero(fgp * fy)
    TN = np.count_nonzero((fgp - 1) * (fy - 1))
    FP = np.count_nonzero(fgp * (fy - 1))
    FN = np.count_nonzero((fgp - 1) * fy)
    return TP, TN, FP, FN

def writeToCSV(info, filename):
    import csv
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["fileHash", "TP", "TN", "FP", "FN"])
        for i in info:
            writer.writerow([str(e) for e in i])

def computeF1(TP, TN, FP, FN):
    if TP + TN == 0:
        precision = -2.0
    else:
        precision = float(TP) / (TP + FP)
    if TP + FN == 0:
        recall = -2.0
    else:
        recall = float(TP) / (TP + FN)
    acc = float(TP+TN)/ (TP+TN+FP+FN)
    f1 = 2 * precision * recall / (precision + recall)
    print("  (precision is: %.5f, recall is: %.5f, f1 is: %.5f)" %
            (precision, recall, f1))

def constructLabelByIndices(y, indices):
    res = []
    for i in range(len(y)):
        numOfNodes = indices[i]
        tmp = [0 for j in range(numOfNodes)]
        tmp[y[i]] = 1
        res.append(tmp)
    return np.concatenate(res)

def computeSEMetric(info, trainFile, filterLabel):
    pred = info[0]
    indices = info[1]
    intraNodeLabel = info[2]
    buggyFileHash = info[3]
    graphPred = info[4]
    y = info[5]
    graphFileHash = info[6]
    #compute clean/buggy prediction results for file level.
    total=[0,0,0,0]
    if filterLabel == 0:
        res = pickResBaseOnFile(graphPred, y, graphFileHash)
        methodInfo = []
        for k in res:
            TP, TN, FP, FN = returnSEMet(res[k])
            total[0] += TP
            total[1] += TN
            total[2] += FP
            total[3] += FN
            methodInfo.append([k, TP, TN, FP, FN])
        if trainFile != "":
            filenameMethodInfo = trainFile
            filenameMethodInfo += "-SEMethod.csv"
            writeToCSV(methodInfo, filenameMethodInfo)
        computeF1(total[0], total[1], total[2], total[3])
    #compute buggy lines.
    else:
        buggyLineInfo = []
        # pred = [1,2,3,1,2,3,4] indices=[3,4] n=1 True
        # [[0, 0, 1], [0, 0, 0, 1]]
        res = returnTopNRes(pred, indices, 1, True)
        res = np.concatenate(res, axis=0)
        res = pickResBaseOnFile(res, intraNodeLabel, buggyFileHash)
        for k in res:
            # useless and wrong
            TP, TN, FP, FN = returnSEMet(res[k])
            buggyLineInfo.append([k, TP, TN, FP, FN])
        if trainFile != "":
            filenameBuggylineInfo = trainFile
            filenameBuggylineInfo += "-SELine.csv"
            writeToCSV(buggyLineInfo, filenameBuggylineInfo)
        for i in [1,3,5,7,10]:
            res = returnTopNRes(pred, indices, i, True)
            res = np.concatenate(res, axis=0)
            nodeLabels = constructLabelByIndices(intraNodeLabel, indices)
            TP, TN, FP, FN = returnSEMet([res, nodeLabels])
            print("top-%d: " % (i))
            computeF1(TP, TN, FP, FN)



def computeTopNAcc(pred, intraNodeLabel, n):
    #res = tf.metrics.mean(tf.nn.in_top_k(predictions=pred, targets=intraNodeLabel, k=n))
    #return res[0]

    _,i = tf.math.top_k(pred, n)
    labels_tiled = tf.tile(tf.expand_dims(intraNodeLabel, axis=-1), [1,n])
    equality = tf.equal(i, labels_tiled)
    logic_or = tf.reduce_any(equality, axis=-1)
    accN = tf.reduce_mean(tf.cast(logic_or, tf.float32))
    return accN

def printTopNAcc(best_accInfo):
    print("  top1 acc is: %.5f, top3 acc is: %.5f, top5 acc is: %.5f, top7 acc is: %.5f, top10 acc is: %.5f." %
            (best_accInfo[0], best_accInfo[1], best_accInfo[2], best_accInfo[3], best_accInfo[4]))

def concatAccInfo(pre, new, num_graphs):
    newRes = []
    if pre == None:
        pre = [[] for i in range(len(new))]
    for i in range(len(pre)):
        newRes.append(list(pre[i])+list(new[i]))
    return newRes


def categoryToIndex(categ):
    assert len(categ) > 0, "bugpos is empty."
    for i in range(len(categ)):
        if categ[i] != 0:
            return i
    return -1

def returnTopNResByThre(preds, indices, threshold=0.0, ifkeep=False):
    pos = 0
    res = []
    for i in range(len(indices)):
        labelInd = indices[i]
        predRes = []
        for predIndex in range(pos, pos+labelInd):
            predRes.append(preds[predIndex])
        pos += labelInd
        sortedIndex = sorted(range(len(predRes)), key=lambda k:predRes[k], reverse = True)
        maxIndex = 0
        for i in range(len(predRes)):
            if predRes[i] > predRes[maxIndex]:
                maxIndex = i
            if predRes[i] > threshold:
                predRes.append(i)
        if len(predRes) == 0:
            predRes.append(maxIndex)

        if ifkeep:
            resArr = [0 for j in range(labelInd)]
            for j in predRes:
                resArr[j] = 1
            predRes = resArr
        res.append(predRes)
    return res

def computeTopNByThre(preds, indices, labels, threshold):
    # preds: [0,1,0,0,1,0,0,1], indices: [2,3,3], labels: [0,2,1]
    res = [0,0]
    pos = 0
    indLength = 0
    for i in indices:
        indLength += i
    assert indLength == len(preds), "Inconsist length"

    predRes = returnTopNResByThre(preds, indices, threshold)

    assert len(predRes) == len(labels), "Inconsist length"

    for i in range(len(predRes)):
        if labels[i] in predRes[i]:
            res[0] += 1
        res[1] += 1

    return float(res[0])/res[1]

def computeTopN(preds, indices, labels, n):
    # preds: [0,1,0,0,1,0,0,1], indices: [2,3,3], labels: [0,2,1]
    res = [0,0]
    pos = 0
    indLength = 0
    for i in indices:
        indLength += i
    assert indLength == len(preds), "Inconsist length"

    predRes = returnTopNRes(preds, indices, n)

    assert len(predRes) == len(labels), "Inconsist length"

    for i in range(len(predRes)):
        if labels[i] in predRes[i]:
            res[0] += 1
        res[1] += 1

    return float(res[0])/res[1]

def computeTopNWithoutInd(preds, labels, n):
    # preds: [[0,1],[0,0,1],[0,0,1], labels: [0,2,1]
    res = [0,0]
    pos = 0
    for i in range(len(preds)):
        predRes = preds[i]
        sortedIndex = sorted(range(len(predRes)), key=lambda k:predRes[k], reverse = True)
        predRes = sortedIndex[:n]
        if labels[i] in predRes:
            res[0] += 1
        res[1] += 1
    return float(res[0])/res[1]

def sortGraphSize(graphs, perc):
    sizesClean = []
    sizesBuggy = []
    for i in range(len(graphs)):
        graph = graphs[i]
        if graph["labels"][0]:
            sizesBuggy.append(len(graph["init"]))
            sizesClean.append(0)
        else:
            sizesClean.append(len(graph["init"]))
            sizesBuggy.append(0)
    return sizesBuggy, sizesClean

def sortGraphSizeInterval(graphs, perc):
    sizesClean = []
    sizesBuggy = []
    for i in range(len(graphs)):
        graph = graphs[i]
        size = 0
        ifBuggy = 0
        for k in graph:
            if k["insideinterval"] == 1:
                size += len(k["init"])
            else:
                ifBuggy = k["labels"][0]

        if ifBuggy:
            sizesBuggy.append(size)
            sizesClean.append(0)
        else:
            sizesClean.append(size)
            sizesBuggy.append(0)
    return sizesBuggy, sizesClean


def filterGraphByPerc(graphs, perc):
    if perc == 100:
        return graphs
    if isinstance(graphs[0], list):
        sizesBuggy, sizesClean = sortGraphSizeInterval(graphs, perc)
    else:
        sizesBuggy, sizesClean = sortGraphSize(graphs, perc)
    assert len(sizesBuggy) == len(sizesClean), "In Filter: inconsist length"

    sortedSizeIndexBuggy = sorted(range(len(sizesBuggy)), key=lambda k:sizesBuggy[k], reverse = True)
    splitIndex = max(1, int(float(perc)/200*len(graphs)))
    sortedSizeIndexBuggy = sortedSizeIndexBuggy[:splitIndex]

    sortedSizeIndexClean = sorted(range(len(sizesClean)), key=lambda k:sizesClean[k], reverse = True)
    splitIndex = max(1, int(float(perc)/200*len(graphs)))
    sortedSizeIndexClean = sortedSizeIndexClean[:splitIndex]
    sortedSizeIndex = sortedSizeIndexBuggy + sortedSizeIndexClean

    graphs[:] = [x for i, x in enumerate(graphs) if i in sortedSizeIndex]


class MLP(object):
    def __init__(self, in_size, out_size, hid_sizes, dropout_keep_prob):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.dropout_keep_prob = dropout_keep_prob
        self.params = self.make_network_params()

    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        weights = [tf.Variable(self.init_weights(s), name='MLP_W_layer%i' % i)
                   for (i, s) in enumerate(weight_sizes)]
        biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name='MLP_b_layer%i' % i)
                  for (i, s) in enumerate(weight_sizes)]

        network_params = {
            "weights": weights,
            "biases": biases,
        }

        return network_params

    def init_weights(self, shape):
        return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (2 * np.random.rand(*shape).astype(np.float32) - 1)

    def __call__(self, inputs):
        acts = inputs
        for W, b in zip(self.params["weights"], self.params["biases"]):
            hid = tf.matmul(acts, tf.nn.dropout(W, self.dropout_keep_prob)) + b
            acts = tf.nn.relu(hid)
        last_hidden = hid
        return last_hidden
