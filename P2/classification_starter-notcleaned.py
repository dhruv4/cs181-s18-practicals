# This file provides starter code for extracting features from xml files and
# for doing some learning.
#
# The basic set-up:
# ----------------
# main() will run code to extract features, learn, and make predictions.
#
# extract_feats() is called by main(), and it will iterate through the
# train/test directories and parse each xml file into an xml.etree.ElementTree,
# which is a standard python object used to represent an xml file in memory.
# (More information about xml.etree.ElementTree objects can be found here:
# http://docs.python.org/2/library/xml.etree.elementtree.html
# and here:
# http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/)
# It will then use a series of "feature-functions" that you will write/modify
# in order to extract dictionaries of features from each ElementTree object.
# Finally, it will produce an N x D sparse design matrix containing the union
# of the features contained in the dicts produced by your "feature-functions."
# This matrix can then be plugged into your learning algorithm.
#
# The learning and prediction parts of main() are largely left to you, though
# it does contain code that randomly picks class-specific weights and predicts
# the class with the weights that give the highest score. If your prediction
# algorithm involves class-specific weights, you should, of course, learn
# these class-specific weights in a more intelligent way.
#
# Feature-functions:
# --------------------
# "feature-functions" are funcs that take an ElementTree object representing
# an xml file (which includes the sequence of system calls a
# piece of potential malware made), and returns a dict mapping feature names to
# their respective numeric values.
# For instance, a simple feature-function might map system call history to the
# dictionary {'first_call-load_image': 1}. This is a boolean feature indicating
# whether the first system call made by the executable was 'load_image'.
# Real-valued or count-based features can also be defined in this way.
# Because this feature-function will be run over ElementTree objects for each
# software execution history instance, we will have the (different)
# feature values of this feature for each history,
# and these values will make up
# one of the columns in our final design matrix.
# Of course, multiple features can be defined within a single dict, and in
# the end all the dictionaries returned by feature functions (for a particular
# training example) will be unioned, so we can collect all the feature values
# associated with that particular instance.
#
# Two example feature-functions, first_last_system_call_feats() and
# system_call_count_feats(), are defined below.
# The first of these functions indicates what the first and last system-calls
# made by an executable are, and the second records the total number of system
# calls made by an executable.
#
# What you need to do:
# --------------------
# 1. Write new feature-functions (or modify the example feature-functions) to
# extract useful features for this prediction task.
# 2. Implement an algorithm to learn from the design matrix produced, and to
# make predictions on unseen data. Naive code for these two steps is provided
# below, and marked by TODOs.
#
# Computational Caveat
# --------------------
# Because the biggest of any of the xml files is only around 35MB,
# the code below
# will parse an entire xml file and store it in memory, compute features, and
# then get rid of it before parsing the next one. Storing the biggest of
# the files
# in memory should require at most 200MB or so, which should be no problem for
# reasonably modern laptops. If this is too much, however, you can lower the
# memory requirement by using ElementTree.iterparse(), which does parsing in
# a streaming way. See
# http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/
# for an example.

import os
import time
import pickle
import json
from BTrees.IIBTree import *
from BTrees.OIBTree import *
from collections import Counter
from random import sample
from datetime import datetime
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# from sknn.mlp import Classifier
# from sknn.mlp import Layer
# from sknn.mlp import Convolution


import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum

from nolearn.lasagne import NeuralNet
from sklearn.externals import joblib

import util

# Two pickled files that we rely on:
#   all-system-calls-classes.pickle,
#       which maps class integers to system call names
#   all-system-calls.pickle,
#       which maps system call names to class integers

GENERATING_SYSTEM_CALL_LIST = False
MAX_SYSTEM_CALLS = 294035

PROPERTIES_PER_CLASS_MULT = 12
CONV1_FILTER_SIZE = 25
CONV1_STRIDE = 1

CONV2_FILTER_SIZE = CONV1_FILTER_SIZE * 2
CONV2_STRIDE = 1


all_system_calls = set()

system_call_codes = None
codes_to_system_calls = None

# code_code_random_mapping = None
code_binary_digits_mapping = {}
time_str = "%M:%S.%f"

GENERATING_FEATURES = {
    "test": False,
    "train": False
}

RUNNING_SAMPLE = False
RUNNING_NNET = False
RUNNING_MLP = True

GENERATING_CODE_BINARY_MAPPING = False

TYPES_SYSTEM_CALLS = 107

RETRAINING_MODEL = True

BINARY_REPR = True

# feature_set_string = "107"
feature_set_string = "bin"


def add_all_system_calls(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    post_condition:
      adds all system calls in this xml document
      to the all_system_calls dict
    """
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            all_system_calls.add(el.tag)


def extract_feats(ffs, direc="train", global_feat_dict=None):
    """
    arguments:
      ffs are a list of feature-functions.
      direc is a directory containing xml files (expected to be train or test).
      global_feat_dict is a dict mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that
      the columns of the test matrix align correctly.

    returns:
      a sparse design matrix, a dict mapping features to column-numbers,
      vector of target classes, and a list of system-call-history ids in order
      of their rows in the design matrix.

      Note: the vector of target classes returned will contain true indices
      of the
      target classes on the training data,
      but will contain only -1's on the test
      data
    """
    fds = []  # list of feature dicts
    classes = []
    ids = []
    if not GENERATING_FEATURES[direc]:
        print "loading features"
        X = sparse.load_npz(
            open("features-%s-%s-X" % (direc, feature_set_string), "rb"))
        feat_dict = json.load(
            open("features-%s-%s-feat-dict.json" % (direc, feature_set_string),
                 "rb"))
        classes = json.load(
            open("features-%s-%s-classes.json" % (direc, feature_set_string),
                 "rb"))
        ids = json.load(
            open("features-%s-%s-ids.json" % (direc, feature_set_string),
                 "rb"))
        print len(feat_dict)
        print 26234 in feat_dict
        print "done loading features"

        return X, feat_dict, np.array(classes), ids

    for datafile in os.listdir(direc):
        # extract id and true class (if available) from filename

        id_str, clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))
        except ValueError:

            # print datafile, clazz

            # should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)
        rowfd = IIBTree()
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc, datafile))

        if GENERATING_SYSTEM_CALL_LIST:
            add_all_system_calls(tree)

        [rowfd.update(ff(tree)) for ff in ffs]
        fds.append(rowfd)

    X, feat_dict = make_design_mat(fds, global_feat_dict)

    print "dumping features"
    sparse.save_npz(
        open("features-%s-%s-X" % (direc, feature_set_string), "wb"), X)
    json.dump(
        feat_dict,
        open("features-%s-%s-feat-dict.json" % (direc, feature_set_string),
             "wb"))
    json.dump(
        classes,
        open("features-%s-%s-classes.json" % (direc, feature_set_string),
             "wb"))
    json.dump(
        ids,
        open("features-%s-%s-ids.json" % (direc, feature_set_string), "wb"))
    print "done dumping features"
    return X, feat_dict, np.array(classes), ids


def make_design_mat(fds, global_feat_dict=None):
    """
    arguments:
      fds is a list of feature dicts (one for each datafile).
      global_feat_dict is a dict mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that
      the columns of the test matrix align correctly.

    returns:
        a sparse NxD design matrix, where N == len(fds) and D is the number of
        the union of features defined in any of the fds
    """
    # generate dict of all possible features
    if global_feat_dict is None:
        all_feats = set()
        [all_feats.update(fd.keys()) for fd in fds]
        all_feats.update(system_call_codes.values())
        [all_feats.add((call + 1) * (-1))
         for call in system_call_codes.values()]
        if BINARY_REPR:
            all_possible_syscall_feats = [
                line * 10 + b
                for line in range(MAX_SYSTEM_CALLS + 1)
                for b in range(((TYPES_SYSTEM_CALLS - 1).bit_length()))
            ]
        else:
            all_possible_syscall_feats = [
                line * 1000 + call_code
                for line in range(MAX_SYSTEM_CALLS + 1)
                for call_code in system_call_codes.values()
            ]
        all_feats.update(all_possible_syscall_feats)
        feat_dict = dict(
            [(feat, i) for i, feat in enumerate(sorted(all_feats))])
    else:
        feat_dict = global_feat_dict

    cols = []
    rows = []
    data = []
    for i in xrange(len(fds)):
        temp_cols = []
        temp_data = []
        for feat, val in fds[i].iteritems():
            try:
                # update temp_cols iff update temp_data
                temp_cols.append(feat_dict[str(feat)])
                temp_data.append(val)
            except KeyError as ex:
                if global_feat_dict is not None:
                    print "new feature in test data this shouldnt happen lol"
                    print feat
                    pass  # new feature in test data; nbd
                else:
                    raise ex

        # all fd's features in the same row
        k = len(temp_cols)
        cols.extend(temp_cols)
        data.extend(temp_data)
        rows.extend([i] * k)

    assert len(cols) == len(rows) and len(rows) == len(data)

    X = sparse.csr_matrix((np.array(data),
                          (np.array(rows), np.array(cols))),
                          shape=(len(fds), len(feat_dict)))
    return X, feat_dict


# Here are two example feature-functions.
# They each take an xml.etree.ElementTree object,
# (i.e., the result of parsing an xml file) and returns a dictionary mapping
# feature-names to numeric values.
# TODO: modify these functions, and/or add new ones.
def first_last_system_call_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'first_call-x' to 1 if x was the first system call
      made, and 'last_call-y' to 1 if y was the last system call made.
      (in other words, it returns a dictionary indicating what the first and
      last system calls made by an executable were.)
    """
    c = Counter()
    in_all_section = False
    first = True  # is this the first system call
    last_call = None  # keep track of last call we've seen
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            if first:
                c["first_call-" + el.tag] = 1
                first = False
            last_call = el.tag  # update last call seen

    # finally, mark last call seen
    c["last_call-" + last_call] = 1
    return c


def harshita_feats(tree):
    # using an int-int optimized tree to store binary features
    # for presence of a system call on a certain line
    # use six features per line that hold a random hash of the system call
    # CALL_LIN * 10 + feature_num[1 to 6] : True or False

    # dict will also contain the counts of specific system calls occuring
    # and the percentage value of count/total_system_calls
    # in the format SYS_CALL_ID: count
    # and -SYS_CALL_ID: count/total_system_calls

    # dict will also contain -(MAX_SYSTEM_CALLS + 1): duration of execution
    features = IIBTree()
    num_calls = 1
    total_duration = 0
    total_calls = 0.0
    global code_binary_digits_mapping

    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "process":
            total_duration += (datetime.strptime(
                el.attrib["terminationtime"], time_str) - datetime.strptime(
                el.attrib["starttime"], time_str)).seconds
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            # found a system call
            total_calls += 1
            call_name = el.tag
            call_code = system_call_codes[call_name]

            # update counts tag
            if call_code not in features:
                features[call_code] = 0
            features[call_code] += 1

            if not BINARY_REPR:
                features[num_calls * 1000 + call_code] = 1
            else:
                for fn, d in enumerate(code_binary_digits_mapping[str(call_code)]):
                        if d:
                            features[num_calls * 10 + fn] = d

            num_calls += 1

    for call in system_call_codes.values():
        if call in features:
                features[(-1) * (call + 1)] = int(features[call] / total_calls)

    features[(MAX_SYSTEM_CALLS + 1) * -1] = total_duration
    # dict.update(c, sys_call_counts)
    # dict.update(c, features)
    # for item in features.items():
    #     print item
    return features


def system_call_count_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'num_system_calls' to the number of system_calls
      made by an executable (summed over all processes)
    """
    # global MAX_SYSTEM_CALLS
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            c['num_system_calls'] += 1
    # MAX_SYSTEM_CALLS = max(c['num_system_calls'], MAX_SYSTEM_CALLS)
    return c


def eval(pred, actual):

    count = 0.0
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            count += 1

    return count / len(actual)


# The following function does the feature extraction, learning, and prediction
def main():
    global MAX_SYSTEM_CALLS
    global system_call_codes
    global codes_to_system_calls
    global h
    global code_code_random_mapping
    global code_binary_digits_mapping

    system_call_codes = pickle.load(open("all-system-calls.pickle", "rb"))
    codes_to_system_calls = pickle.load(
        open("all-system-calls-classes.pickle", "rb"))

    if GENERATING_CODE_BINARY_MAPPING:
        code_binary_digits_mapping = {}
        code_code_random_mapping = {k: v for k, v in zip(
            sample(range(0, TYPES_SYSTEM_CALLS), TYPES_SYSTEM_CALLS),
            sample(range(0, 2 ** (TYPES_SYSTEM_CALLS - 1).bit_length()),
                   TYPES_SYSTEM_CALLS))}

        for code in range(0, TYPES_SYSTEM_CALLS):
            randomized_code = code_code_random_mapping[code]
            result = []
            for char in format(randomized_code,
                               "0" + str(
                                   (TYPES_SYSTEM_CALLS - 1).bit_length()) +
                               "b"):
                result.append(int(char))
            code_binary_digits_mapping[code] = result
        json.dump(code_binary_digits_mapping,
                  open("code-binary-ft-map.json", "wb"))
    else:
        code_binary_digits_mapping = json.load(
            open("code-binary-ft-map.json", "rb"))

    train_dir = "train"
    test_dir = "test"

    outputfile = "sample_predictions%s.csv" % time.strftime("%Y%m%d-%H%M%S")
    outputfilemlp = "rfr_predictions%s.csv" % time.strftime("%Y%m%d-%H%M%S")
    outputfilenet1 = "net1_predictions%s.csv" % time.strftime("%Y%m%d-%H%M%S")

    ffs = [harshita_feats]

    # extract features
    if True:
        print "extracting training features..."
        X_train, global_feat_dict, t_train, train_ids = extract_feats(
            ffs, train_dir)
        print "done extracting training features"
        print
        X_train, X_valid, t_train, t_valid = train_test_split(
            X_train, t_train, test_size=0.33)


    # print X_train X_train is the set of tuples
    # with features to values (id, feature_id): value_for_feature

    # print global_feat_dict global_feat_dict
    # relates the feature_id to the value_for_feature

    # print t_train is a vector of the actual classes for each X_train id

    # print global_feat_dict
    print "learning..."
    if RUNNING_SAMPLE:
        learned_W = np.random.random(
            (len(global_feat_dict), len(util.malware_classes)))
    if RUNNING_MLP:
        mlpclf = RandomForestClassifier(
            verbose=5,
            n_jobs=-1,
            max_features=0.95,
            n_estimators=400,
            oob_score=True
        )
        # # mlpclf = MLPClassifier(
        #     verbose=True,
        #     hidden_layer_sizes=(
        #         len(util.malware_classes) * PROPERTIES_PER_CLASS_MULT,),
        #         # len(util.malware_classes) * PROPERTIES_PER_CLASS_MULT / 2,),
        #     learning_rate="adaptive",
        #     early_stopping=True)

    if RUNNING_NNET:
        net1 = NeuralNet(
            layers=[(layers.InputLayer, {
                    "shape": (None, X_train.shape[0], X_train.shape[1])}),
                    (layers.Conv1DLayer, {
                        "num_filters": len(
                            util.malware_classes) * PROPERTIES_PER_CLASS_MULT,
                        "filter_size": CONV1_FILTER_SIZE,
                        "stride": CONV1_STRIDE,
                        "untie_biases": True,
                        "nonlinearity": lasagne.nonlinearities.sigmoid,
                        "W": lasagne.init.GlorotUniform()
                    }),
                    # ('maxpool1', layers.MaxPool1DLayer),
                    (layers.Conv1DLayer, {
                        "num_filters": len(
                            util.malware_classes) * PROPERTIES_PER_CLASS_MULT,
                        "filter_size": CONV2_FILTER_SIZE,
                        "stride": CONV2_STRIDE,
                        "pad": "full",
                        "untie_biases": True,
                        "nonlinearity": lasagne.nonlinearities.sigmoid,
                        "W": lasagne.init.GlorotUniform()
                    }),
                    # ('maxpool2', layers.MaxPool2DLayer),
                    (layers.DropoutLayer, {"p": 0.5}),
                    (layers.DenseLayer, {
                        "num_units": 256,
                        "nonlinearity": lasagne.nonlinearities.sigmoid}),
                    (layers.DropoutLayer, {"p": 0.5}),
                    (layers.DenseLayer, {
                        "nonlinearity": lasagne.nonlinearities.sigmoid,
                        "num_units": len(util.malware_classes)}),
                    ],
            # optimization method params
            update=nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.9,
            max_epochs=10,
            verbose=1,
        )

    # tt = X_train.toarray().astype(int)

    if RUNNING_NNET:
        net1.fit(tt.reshape(tt.shape + (1,)), t_train.astype(np.int32))
    if RUNNING_MLP:
        if RETRAINING_MODEL:
            mlpclf.fit(X_train, t_train)
            joblib.dump(mlpclf, "classifier-model.pickle")

            print "done learning"
            print

            print "validation testing"
            if RUNNING_MLP:
                validmlp = mlpclf.predict(X_valid.toarray())
                print "mlp prediction", validmlp
                print "t actual", t_valid
                print eval(validmlp, t_valid)

            if RUNNING_NNET:
                validnet1 = net1.predict(X_valid.toarray())
                print "nnet prediction", validnet1
                print "t actual", t_valid
                print eval(validnet1, t_valid)

        else:
            mlpclf = joblib.load("classifier-model.pickle")
            print "loaded trained model, not validating"

    # get rid of training data and load test data
    del X_train
    # del tt
    del t_train
    del train_ids

    print "extracting test features..."
    X_test, _, t_ignore, test_ids = extract_feats(
        ffs, test_dir, global_feat_dict=global_feat_dict)
    print "done extracting test features"
    print

    print "making predictions..."


    if RUNNING_SAMPLE:
        preds = np.argmax(X_test.dot(learned_W), axis=1)

    if RUNNING_MLP:
        predsmlp = mlpclf.predict(X_test)
        print predsmlp

    if RUNNING_NNET:
        predsnet1 = net1.predict(X_test.toarray())

    print "done making predictions"
    print

    print "writing predictions..."

    if RUNNING_SAMPLE:
        util.write_predictions(preds, test_ids, outputfile)

    if RUNNING_MLP:
        util.write_predictions(predsmlp, test_ids, outputfilemlp)

    if RUNNING_NNET:
        util.write_predictions(predsnet1, test_ids, outputfilenet1)

    print "done!"

    print MAX_SYSTEM_CALLS

    if GENERATING_SYSTEM_CALL_LIST:
        pickle.dump(all_system_calls, open("all-system-calls.pickle", 'wb'))


if __name__ == "__main__":
    main()
