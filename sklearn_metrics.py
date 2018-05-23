#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
from collections import defaultdict
import numpy as np
import sklearn.metrics as metrics


def relative_error_in_total_energy(target, predict):
    """Negative means under-estimates."""
    sum_output = np.sum(predict)
    sum_target = np.sum(target)
    return float(
        (sum_output - sum_target) / (max(sum_output, sum_target)+0.01))

METRICS = {
    'classification': {
        'accuarcy' : metrics.accuracy_score,
        'F1_score' : metrics.f1_score,
        'precision' : metrics.precision_score,
        'recall' : metrics.recall_score,
        'confusion_matrix' : metrics.confusion_matrix
        },
    'regression': {
        'mae' : metrics.mean_absolute_error,
        'mse' : metrics.mean_squared_error,
        'energy_metrics' : relative_error_in_total_energy
    }
}


class Metrics():
    def __init__(self, APPLIANCES, target_series, pred_series, acc_threshold = 15):
        self.chk_input_data(target_series, pred_series)
        self.init_parameters(APPLIANCES, target_series, pred_series, acc_threshold)
        self.init_dataStructure()
        self.init_matrics_dict()

    def chk_input_data(self, target_series, pred_series):
        if target_series.shape != pred_series.shape:
            raise ValueError("output.shape != target.shape")

    def init_parameters(self, APPLIANCES, target_series, pred_series, acc_threshold):
        self.target = target_series.flatten()
        self.pred = pred_series.flatten()

        self.target_on_off = np.array([self.target >= acc_threshold]).flatten()
        self.pred_on_off = np.array([self.pred >= acc_threshold]).flatten()

        self.appliances = APPLIANCES
        self.NUM_SEQ_PER_BATCH = target_series.shape[0]
        self.num_appliance = target_series.shape[1]
        self.seq_length = target_series.shape[2]
        self.acc_threshold = acc_threshold

    def init_dataStructure(self):
        self.acc_metrics = defaultdict(lambda: 0)
        self.F1_score = defaultdict(lambda: 0)
        self.precision = defaultdict(lambda: 0)
        self.recall = defaultdict(lambda: 0)
        self.energy_metrics = defaultdict(lambda: 0)
        self.mse = defaultdict(lambda: 0)
        self.mae = defaultdict(lambda: 0)
        self.CF = defaultdict(lambda: [0, 0, 0, 0])

    def init_matrics_dict(self):
        self.classification = {
            'accuarcy' : self.acc_metrics,
            'F1_score' : self.F1_score,
            'precision' : self.precision,
            'recall' : self.recall,
            'confusion_matrix' : self.CF
        }

        self.regression = {
            'mse' : self.mse,
            'mae' : self.mae,
            'energy_metrics' : self.energy_metrics
        }

        self.metrics = {
            'classification' : self.classification,
            'regression' : self.regression
        }

    def compute_metrics(self):
        for batch_idx in range(self.NUM_SEQ_PER_BATCH - 1):
            batch_start = batch_idx * self.seq_length * self.num_appliance
            batch_end = (batch_idx + 1) * self.seq_length * self.num_appliance

            target_classification_batch_series = self.target_on_off[batch_start:batch_end]
            pred_classification_batch_series = self.pred_on_off[batch_start:batch_end]

            target_regression_batch_series = self.target[batch_start:batch_end]
            pred_regression_batch_series = self.pred[batch_start:batch_end]

            for appliance_idx in range(self.num_appliance):
                self.compute_appliance_metrics(target_classification_batch_series, pred_classification_batch_series, appliance_idx, 'classification')
                self.compute_appliance_metrics(target_regression_batch_series, pred_regression_batch_series, appliance_idx, 'regression')

    def compute_appliance_metrics(self, target_batch_series, pred_batch_series, appliance_idx, category):
        label = self.appliances[appliance_idx]
        appliance_start = self.seq_length * appliance_idx
        appliance_end = self.seq_length * (appliance_idx + 1)

        target_series = target_batch_series[appliance_start:appliance_end]
        pred_series = pred_batch_series[appliance_start:appliance_end]

        for metrics_type, metrics in METRICS[category].iteritems():
            result = metrics(target_series, pred_series)

            if metrics_type == 'confusion_matrix':
                result = result.flatten()
                self.metrics[category][metrics_type][label] += result
            else:
                self.metrics[category][metrics_type][label] += result / self.NUM_SEQ_PER_BATCH

    def print_metrics(self):
        for appliance_idx in range(self.num_appliance):
            label = self.appliances[appliance_idx]
            print('*' * 25)
            for category in METRICS.keys():
                print('-'*5, 'validation of', category, 'metrics', '-'*5)

                for metrics_type in METRICS[category].keys():
                    if metrics_type == 'confusion_matrix':
                        print('TN of', label, '=', self.metrics[category][metrics_type][label][0])
                        print('FP of', label, '=', self.metrics[category][metrics_type][label][1])
                        print('FN of', label, '=', self.metrics[category][metrics_type][label][2])
                        print('TP of', label, '=', self.metrics[category][metrics_type][label][3])
                    else:
                        print(metrics_type, 'of', label, '= {:.3f}'.format(self.metrics[category][metrics_type][label]))

                print('')