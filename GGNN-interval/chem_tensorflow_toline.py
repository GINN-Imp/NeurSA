#!/usr/bin/env/python
"""
Usage:
    chem_tensorflow_sparse.py [options]

Options:
    -h --help                Show this screen.
    --config-file FILE       Hyperparameter configuration file path (in JSON format).
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format).
    --log_dir DIR            Log dir name.
    --data_dir DIR           Data dir name.
    --restore FILE           File to restore weights from.
    --freeze-graph-model     Freeze weights of graph model components.
"""
from typing import List, Tuple, Dict, Sequence, Any

from docopt import docopt
from collections import defaultdict, namedtuple
import numpy as np
import tensorflow as tf
import sys, traceback, os, json
import pdb
import itertools

from chem_tensorflow_interval import IntervalGGNNChemModel
from utils import MLP, glorot_init, SMALL_NUMBER, computeTopN
import utils

from tensor2tensor.models import transformer
from tensor2tensor.data_generators import problem_hparams



GGNNWeights = namedtuple('GGNNWeights', ['edge_weights',
                                         'edge_biases',
                                         'edge_type_attention_weights',
                                         'rnn_cells',])


class IntervalGGNNChemModelLine(IntervalGGNNChemModel):
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            #'batch_size': 100000,
            'batch_size': 200,
            'rep_mode': 0,
            'toLine': True,
            #'train_file': 'intervals-jfreechart-1.0.19.json',
            'train_file': '/proj/fff000/ggnnbl/spoon-intervals/jsondata/intervals-projects-defects4j-train.json',
            'valid_file': '/proj/fff000/ggnnbl/spoon-intervals/jsondata/intervals-projects-defects4j-train.json',
            'use_edge_bias': False,
            'accInfoPreprocess': False,
            'debugAccInfo': False,
            'threeEmbedding': False,
            'filterLabel': 0,
            'outputCSVPrefix': "",
            'use_propagation_attention': False,
            'use_edge_msg_avg_aggregation': True,
            'residual_connections': {  # For layer i, specify list of layers whose output is added as an input
                                     "2": [0],
                                     "4": [0, 2]
                                    },

            'layer_timesteps': [2, 2, 1, 2, 1],  # number of layers & propagation steps per layer

            'graph_rnn_cell': 'GRU',  # GRU, CudnnCompatibleGRUCell, or RNN
            'graph_rnn_activation': 'tanh',  # tanh, ReLU
            'graph_state_dropout_keep_prob': 1.,
            'task_sample_ratios': {},
            'iterSteps': 3,
            'filterTrain': False,
            'edge_weight_dropout_keep_prob': .8
        })
        return params

    def load_data(self, file_name, is_training_data: bool):
        processed_graphs = IntervalGGNNChemModel.load_data(self, file_name, is_training_data)
        if self.params['filterLabel'] != 0:
            processed_graphs[:] = [x for x in processed_graphs if self.isBuggyMethod(x)]
        if is_training_data == self.params["filterTrain"]:
            utils.filterGraphByPerc(processed_graphs, self.params['on_large_data'])
        print("after filtering, remain %d graphs."%(len(processed_graphs)))
        return processed_graphs

    def computeLoss(self, task_id, internal_id):
        embeddingNum = 2
        if self.params['threeEmbedding'] == True:
            embeddingNum = 3
        numOfOutput = 1

        graphPred = self.computePred(task_id, internal_id)
        y = tf.to_int64(self.placeholders['target_values'][internal_id,:])
        graphCost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=graphPred, labels=y)
        graphPred = tf.argmax(graphPred,1)
        correct_pred = tf.equal(graphPred, y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        if self.params['debugAccInfo']:
            self.ops['accuracy_task%i' % task_id] = tf.constant(0)
        else:
            self.ops['accuracy_task%i' % task_id] = accuracy


        if embeddingNum == 3:
            #TODO: this is incorrect.
            self.computeFinalNodeRepresentaion(graphPred)


        if self.params["rep_mode"] == 4 and self.params["filterLabel"] == 1:
            with tf.variable_scope("regression_gate"):
                self.weights['regression_gate_task%i' % task_id] = MLP(self.params['hidden_size'], numOfOutput,
                        [], self.placeholders['out_layer_dropout_keep_prob'])
            with tf.variable_scope("regression"):
                self.weights['regression_transform_task%i' % task_id] = MLP(self.params['hidden_size'], numOfOutput,
                        [], self.placeholders['out_layer_dropout_keep_prob'])
            nodePred = self.useAttention(self.weights['regression_gate_task%i' % task_id],
                    self.weights['regression_transform_task%i' % task_id])
        else:

            with tf.variable_scope("regression_gate"):
                self.weights['regression_gate_task%i' % task_id] = MLP(embeddingNum * self.params['hidden_size'], numOfOutput,
                        [], self.placeholders['out_layer_dropout_keep_prob'])
            with tf.variable_scope("regression"):
                self.weights['regression_transform_task%i' % task_id] = MLP(self.params['hidden_size'], numOfOutput,
                        [], self.placeholders['out_layer_dropout_keep_prob'])
            nodePred = self.gated_regression_for_line(self.ops['final_node_representations'],
                    self.weights['regression_gate_task%i' % task_id],
                    self.weights['regression_transform_task%i' % task_id])

        nodePred = tf.squeeze(nodePred)

        #intraY = tf.to_int64(self.placeholders['intraIntervalNodeLabels'])
        #cost = tf.losses.hinge_loss(labels=intraY, logits=pred)
        intraY = tf.to_float(self.placeholders['intraIntervalNodeLabels'])
        intraLabelMasks = self.placeholders['intraLabelMasks']

        intraY = tf.boolean_mask(intraY, intraLabelMasks)

        if self.params['accInfoPreprocess']:
            self.returnTopNAcc(nodePred, intraLabelMasks, task_id)
        else:
            self.preprocessLabels(nodePred, intraLabelMasks, task_id, graphPred, y)
            acc1 = self.returnTopNAcc(nodePred, intraLabelMasks, task_id)

        nodePred = tf.boolean_mask(nodePred, intraLabelMasks)
        nodeCost = tf.nn.sigmoid_cross_entropy_with_logits(labels=intraY, logits=nodePred)
        nodeCost = tf.reduce_sum(nodeCost)
        graphCost = tf.reduce_sum(graphCost)
        #cost = tf.stack([nodeCost, graphCost])
        if self.params['filterLabel'] != 0:
            self.ops['accuracy_task%i' % task_id] = acc1
            cost = nodeCost
        else:
            cost = graphCost
        self.ops['losses'].append(cost)

    def preprocessLabels(self, pred, intraLabelMasks, task_id, graphPred, y):
        intraNodeLabel = self.placeholders['intraIntervalNodeLabelsIndex']
        intraLabelIndex = self.placeholders['intraLabelIndex']
        indices = self.placeholders['numOfNodesInGraph']
        graphLabelMasks = self.placeholders['graphLabelMasks']
        pred = tf.boolean_mask(pred, intraLabelMasks)
        intraNodeLabel = tf.boolean_mask(intraNodeLabel, graphLabelMasks)
        indices = tf.boolean_mask(indices, graphLabelMasks)

        graphFileHash = self.placeholders['fileHash']
        buggyFileHash = tf.boolean_mask(graphFileHash, graphLabelMasks)
        self.ops['acc_info'] = [pred, indices, intraNodeLabel, buggyFileHash, graphPred, y, graphFileHash]


    def processAccInfo(self, pre, new, num_graphs):
        if self.params['accInfoPreprocess']:
            if self.params['debugAccInfo']:
                return self.concatAccInfo(pre, new, num_graphs)
            else:
                return self.computeAccuracy(pre, new, num_graphs)
        else:
            return utils.concatAccInfo(pre, new, num_graphs)

    def computeAccuracy(self, pre, new, num_graphs):
        newRes = []
        if pre == None:
            pre = [[],[],[],[], 0]
        for i in range(len(pre)-1):
            pre[i].append(np.array(new[i])*num_graphs)
        pre[-1] += num_graphs
        return pre

    def returnTopNAcc(self, pred, intraLabelMasks, task_id):
        intraNodeLabel = self.placeholders['intraIntervalNodeLabelsIndex']
        # e.g., intraLabelIndex: [[0,1,2],[3],[4,5,6,7]]; pred: [1,2,3,4,5,6,7,8]
        # => intraLabelIndex: [[0,1,2,8],[3,8,8,8],[4,5,6,7]]; pred: [1,2,3,4,5,6,7,8,-inf]
        intraLabelIndex = self.placeholders['intraLabelIndex']
        #pred = tf.to_float(self.placeholders['intraIntervalNodeLabels'])
        pred = tf.concat([pred,[float("-inf")]], axis=0)
        # lookup=>[[1,2,3,-1], [4,-1,-1,-1], [5,6,7,8]]
        pred = tf.nn.embedding_lookup(pred, intraLabelIndex)
        graphLabelMasks = self.placeholders['graphLabelMasks']
        pred = tf.boolean_mask(pred, graphLabelMasks)
        intraNodeLabel = tf.boolean_mask(intraNodeLabel, graphLabelMasks)
        # set negative values to 0
        #pred = tf.nn.relu(pred)
        return self.computeTopNAcc(pred, intraNodeLabel, 1)

        if self.params['debugAccInfo']:
            self.ops['acc_info'] = [pred, intraNodeLabel, tf.constant(0, shape=[1]), tf.constant(0, shape=[1])]
        else:
            acc1 = self.computeTopNAcc(pred, intraNodeLabel, 1)
            acc2 = self.computeTopNAcc(pred, intraNodeLabel, 3)
            acc3 = self.computeTopNAcc(pred, intraNodeLabel, 5)
            acc5 = self.computeTopNAcc(pred, intraNodeLabel, 10)
            self.ops['acc_info'] = [acc1, acc2, acc3, acc5]

    def computeTopNAcc(self, pred, intraNodeLabel, n):
        return utils.computeTopNAcc(pred, intraNodeLabel, n)

    def parseAccInfoWhenAcc(self, accInfo):
        best_accInfo = []
        for i in range(len(accInfo)-1):
            best_accInfo.append(np.sum(accInfo[i], axis=0)/accInfo[-1])
        return best_accInfo


    def parseAccInfo(self, best_accInfo):
        if self.params['accInfoPreprocess']:
            if self.params['debugAccInfo']:
                pred = best_accInfo[0]
                intraNodeLabel = best_accInfo[1]
                best_accInfo = [0,0,0,0,0]
                best_accInfo[0] = utils.computeTopNWithoutInd(pred, intraNodeLabel, 1)
                best_accInfo[1] = utils.computeTopNWithoutInd(pred, intraNodeLabel, 3)
                best_accInfo[2] = utils.computeTopNWithoutInd(pred, intraNodeLabel, 5)
                best_accInfo[3] = utils.computeTopNWithoutInd(pred, intraNodeLabel, 7)
                best_accInfo[4] = utils.computeTopNWithoutInd(pred, intraNodeLabel, 10)
                utils.printTopNAcc(best_accInfo)
                return
            else:
                # if computeTopNAcc is used
                best_accInfo = self.parseAccInfoWhenAcc(best_accInfo)
                utils.printTopNAcc(best_accInfo)
                return
        else:
            # if preprocessLabels is used
            pred = best_accInfo[0]
            indices = best_accInfo[1]
            intraNodeLabel = best_accInfo[2]
            acc = [0,0,0,0,0]
            acc[0] = computeTopN(pred, indices, intraNodeLabel, 1)
            acc[1] = computeTopN(pred, indices, intraNodeLabel, 3)
            acc[2] = computeTopN(pred, indices, intraNodeLabel, 5)
            acc[3] = computeTopN(pred, indices, intraNodeLabel, 7)
            acc[4] = computeTopN(pred, indices, intraNodeLabel, 10)
            if self.params['filterLabel'] != 0:
                utils.printTopNAcc(acc)
            #if self.params["outputCSVPrefix"] != "":
            utils.computeSEMetric(best_accInfo, self.params["outputCSVPrefix"], self.params['filterLabel'])

    def useAttention(self, regression_gate, regression_transform):
        finalGraphRep = self.ops["final_node_representations"]
        finalGraphRep = tf.unsorted_segment_sum(data=finalGraphRep,
                segment_ids=self.ops['nodeListsForGated'],
                num_segments=self.ops['numGraphForGated'])  # [g x hidden_size]
        if self.ops.get('nodeListsForGated2') != None:
            finalGraphRep = tf.unsorted_segment_sum(data=finalGraphRep,
                    segment_ids=self.ops['nodeListsForGated2'],
                    num_segments=self.ops['numGraphForGated2'])  # [g x hidden_size]
        finalGraphRep = tf.nn.embedding_lookup(finalGraphRep, self.placeholders['intervalGraphNodesList'])
        finalGraphRep = tf.nn.embedding_lookup(finalGraphRep, self.placeholders['graph_nodes_list'])

        x = self.placeholders['converted_node_representation']
        embeddings = tf.get_variable('embedding_matrix', [self.number_of_tokens, self.params["hidden_size"]])
        x = tf.nn.embedding_lookup(embeddings, x) # batch_size*maxWidth*hidden_size
        # Get the number of rows in the fed value at run-time.
        ph_num_rows = tf.shape(x)[1]
        finalGraphRep = tf.expand_dims(finalGraphRep, axis=1) # Add dimension
        finalGraphRep = tf.tile(finalGraphRep, multiples=tf.stack([1,ph_num_rows,1])) # Duplicate in this dimension
        con_x = tf.concat([x,finalGraphRep], axis=-1) # Concatenate on innermost dimension


        with tf.variable_scope("feedforward"):
            weights = {
                    'out': tf.Variable(tf.random_normal([2*self.params['hidden_size'], 1]))
                    }
            biases = {
                    'out': tf.Variable(tf.random_normal([1]))
                    }

        C = tf.tensordot(con_x, weights['out'], axes=1) + biases['out'] # batch_size*maxWidth*1
        C = tf.squeeze(C)
        C = tf.nn.softmax(C)
        C = tf.expand_dims(C, 2)
        C = tf.math.multiply(C, x) # batch_size*maxwidth*hidden_size
        C = tf.reduce_sum(C, 2) # batch_size*maxwidth

        #gate_input = tf.concat([C, self.placeholders['initial_node_representation']], axis=-1)  # [v x 2h]
        #gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)  # [v x 1]
        gated_outputs = regression_gate(C) * regression_transform(C)  # [v x 1]
        return gated_outputs

    def gated_regression_for_line(self, last_h, regression_gate, regression_transform):
        # last_h: [v x h]
        if self.params['threeEmbedding']:
            gate_input = tf.concat([last_h, self.placeholders['initial_node_representation'], self.ops["finalGraphRepresentation"]], axis=-1)  # [v x 3h]
        else:
            gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=-1)  # [v x 2h]
        #gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)  # [v x 1]
        gated_outputs = regression_gate(gate_input) * regression_transform(last_h)  # [v x 1]

        return gated_outputs  # [g]

    def computeFinalNodeRepresentaion(self, pred):
        finalGraphRep = tf.nn.embedding_lookup(pred, self.placeholders['intervalGraphNodesList'])
        finalGraphRep = tf.nn.embedding_lookup(finalGraphRep, self.placeholders['graph_nodes_list'])
        self.ops["finalGraphRepresentation"] = finalGraphRep

def main():
    args = docopt(__doc__)
    try:
        model = IntervalGGNNChemModelLine(args)
        model.train()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    main()
