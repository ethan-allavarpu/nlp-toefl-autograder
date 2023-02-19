#!/usr/bin/env python
'''
Simple eval script that doesn't rely on external libraries

@author: Dan Blanchard (dblanchard@ets.org)
@date: February, 2013
'''

from __future__ import print_function, unicode_literals

import argparse
import sys
from collections import defaultdict
from itertools import izip
from operator import itemgetter


#### BEGIN CODE TAKEN FROM Natural Language Toolkit https://github.com/nltk/nltk/blob/master/nltk/metrics/scores.py ####
def accuracy(reference, test):
    """
    Given a list of reference values and a corresponding list of test
    values, return the fraction of corresponding values that are
    equal.  In particular, return the fraction of indices
    ``0<i<=len(test)`` such that ``test[i] == reference[i]``.

    :type reference: list
    :param reference: An ordered list of reference values.
    :type test: list
    :param test: A list of values to compare against the corresponding
        reference values.
    :raise ValueError: If ``reference`` and ``length`` do not have the
        same length.
    """
    if len(reference) != len(test):
        raise ValueError("Lists must have the same length.")
    return float(sum(x == y for x, y in izip(reference, test))) / len(test)


def precision(reference, test):
    """
    Given a set of reference values and a set of test values, return
    the fraction of test values that appear in the reference set.
    In particular, return card(``reference`` intersection ``test``)/card(``test``).
    If ``test`` is empty, then return None.

    :type reference: set
    :param reference: A set of reference values.
    :type test: set
    :param test: A set of values to compare against the reference set.
    :rtype: float or None
    """
    if (not hasattr(reference, 'intersection') or
        not hasattr(test, 'intersection')):
        raise TypeError('reference and test should be sets')

    if len(test) == 0:
        return None
    else:
        return float(len(reference.intersection(test))) / len(test)


def recall(reference, test):
    """
    Given a set of reference values and a set of test values, return
    the fraction of reference values that appear in the test set.
    In particular, return card(``reference`` intersection ``test``)/card(``reference``).
    If ``reference`` is empty, then return None.

    :type reference: set
    :param reference: A set of reference values.
    :type test: set
    :param test: A set of values to compare against the reference set.
    :rtype: float or None
    """
    if (not hasattr(reference, 'intersection') or
        not hasattr(test, 'intersection')):
        raise TypeError('reference and test should be sets')

    if len(reference) == 0:
        return None
    else:
        return float(len(reference.intersection(test))) / len(reference)


def f_measure(reference, test, alpha=0.5):
    """
    Given a set of reference values and a set of test values, return
    the f-measure of the test values, when compared against the
    reference values.  The f-measure is the harmonic mean of the
    ``precision`` and ``recall``, weighted by ``alpha``.  In particular,
    given the precision *p* and recall *r* defined by:

    - *p* = card(``reference`` intersection ``test``)/card(``test``)
    - *r* = card(``reference`` intersection ``test``)/card(``reference``)

    The f-measure is:

    - *1/(alpha/p + (1-alpha)/r)*

    If either ``reference`` or ``test`` is empty, then ``f_measure``
    eturns None.

    :type reference: set
    :param reference: A set of reference values.
    :type test: set
    :param test: A set of values to compare against the reference set.
    :rtype: float or None
    """
    p = precision(reference, test)
    r = recall(reference, test)
    if p is None or r is None:
        return None
    if p == 0 or r == 0:
        return 0
    return 1.0 / (alpha / p + (1 - alpha) / r)
#### END CODE TAKEN FROM NLTK ####


def get_stat_string(class_result_dict, stat):
    ''' Little helper for getting output for precision, recall, and f-score columns in confusion matrix. '''
    if stat in class_result_dict and class_result_dict[stat] is not None:
        return "{0:.1f}%".format(class_result_dict[stat] * 100)
    else:
        return "N/A"


if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Takes two CSV files, and generates per-class precision, recall, and f1 scores.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('actualfile', help='File containing the actual/gold-standard values.', type=argparse.FileType('r'))
    parser.add_argument('predictedfile', help='File containing the predictions from a classifier.', type=argparse.FileType('r'))
    parser.add_argument('-d', '--delimiter', help='Delimiter separating fields', default=",")
    parser.add_argument('-a', '--actual_field', help='The field in ACTUALFILE to use as the  class value.' +
                                                     'Note: Values are indexed from 1 to be similar to the "cut" command.',
                        default=3, type=int)
    parser.add_argument('-i', '--id_field', help='The field in both files that contains the unique ID of each instance.' +
                                                 'Note: Values are indexed from 1 to be similar to the "cut" command.', default=1, type=int)
    parser.add_argument('-p', '--predicted_field', help='The field in PREDICTEDFILE to use as the class value.' +
                                                        'Note: Values are indexed from 1 to be similar to the "cut" command.',
                        default=2, type=int)
    parser.add_argument('-m', '--conf_matrix', help='Generate a confusion matrix and more nicely formatted output. Requires scikit-learn and Texttable.', action='store_true')
    args = parser.parse_args()

    # Adjust field value to be like a real array index
    args.actual_field -= 1
    args.predicted_field -= 1
    args.id_field -= 1

    # Read values
    actual_tuples = sorted((line.strip().split(args.delimiter) for line in args.actualfile if line.strip()), key=itemgetter(args.id_field))
    predicted_tuples = sorted((line.strip().split(args.delimiter) for line in args.predictedfile if line.strip()), key=itemgetter(args.id_field))

    # Check that lists are the same length
    if len(actual_tuples) != len(predicted_tuples):
        print("ERROR: Number of predictions ({0}) does not match number of actual values ({1}).".format(len(predicted_tuples), len(actual_tuples)), file=sys.stderr)
        sys.exit(2)

    actual_id_set = set(tup[args.id_field] for tup in actual_tuples)
    pred_id_set = set(tup[args.id_field] for tup in predicted_tuples)

    # Check if there are IDs in file that are not in the other
    extra_ids = pred_id_set - actual_id_set
    missing_ids = actual_id_set - pred_id_set
    if extra_ids:
        print("ERROR: There are extra IDs present in {0} that are not in {1}: {2}".format(args.predictedfile.name, args.actualfile.name, ', '.join(sorted(extra_ids))),
              file=sys.stderr)
    if missing_ids:
        print("ERROR: There are IDs present in {1} that are missing from {0}: {2}".format(args.predictedfile.name, args.actualfile.name, ', '.join(sorted(missing_ids))),
              file=sys.stderr)
    if extra_ids or missing_ids:
        sys.exit(2)

    # Compute and store other metrics
    actual_dict = defaultdict(set)
    pred_dict = defaultdict(set)
    actual = []
    predicted = []
    for pred_tuple, actual_tuple in izip(predicted_tuples, actual_tuples):
        pred_class = pred_tuple[args.predicted_field]
        pred_id = pred_tuple[args.id_field]
        actual_class = actual_tuple[args.actual_field]
        actual_id = actual_tuple[args.id_field]
        pred_dict[pred_class].add(pred_id)
        actual_dict[actual_class].add(actual_id)
        actual.append(actual_class)
        predicted.append(pred_class)
    classes = sorted(set(actual_dict.keys() + pred_dict.keys()))
    result_dict = defaultdict(dict)
    for class_name in classes:
        result_dict[class_name]["Precision"] = precision(actual_dict[class_name], pred_dict[class_name])
        result_dict[class_name]["Recall"] = recall(actual_dict[class_name], pred_dict[class_name])
        result_dict[class_name]["F-measure"] = f_measure(actual_dict[class_name], pred_dict[class_name])
    accuracy = accuracy(actual, predicted) * 100

    # Print fancy output with confusion matrix if asked
    if args.conf_matrix:
        from sklearn import metrics
        from texttable import Texttable

        # Compute confusion matrix
        conf_matrix = metrics.confusion_matrix(actual, predicted).tolist()

        # Print confusion matrix with precision, recall, and f-measure for all classes
        result_table = Texttable(max_width=0)
        result_table.set_cols_align(["r"] * (len(classes) + 4))
        result_table.add_rows([[""] + classes + ["Precision", "Recall", "F-measure"]], header=True)
        for i, actual_class in enumerate(classes):
            conf_matrix[i][i] = "[{0}]".format(conf_matrix[i][i])
            class_prec = get_stat_string(result_dict[actual_class], "Precision")
            class_recall = get_stat_string(result_dict[actual_class], "Recall")
            class_f = get_stat_string(result_dict[actual_class], "F-measure")
            result_table.add_row([actual_class] + conf_matrix[i] + [class_prec, class_recall, class_f])
        print(result_table.draw())
        print("(row = reference; column = predicted)")
        print("Accuracy = {0:.1f}%\n".format(accuracy))

    # Otherwise print simple output
    else:
        print("Class\tPrec.\tRec.\tF1")
        for actual_class in classes:
            class_prec = get_stat_string(result_dict[actual_class], "Precision")
            class_recall = get_stat_string(result_dict[actual_class], "Recall")
            class_f = get_stat_string(result_dict[actual_class], "F-measure")
            print('\t'.join([actual_class, class_prec, class_recall, class_f]))
        print("\nOverall Accuracy = {0:.1f}%\n".format(accuracy))
