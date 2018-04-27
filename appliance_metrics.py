#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

class metric_calculation():
    def __init__(self, MULTI_APPLIANCE, NUM_SEQ_PER_BATCH, seq_length, target_series, pred_series, acc_threshold = 25, series = 5):
        self.target = target_series
        self.pred = pred_series
        self.appliances = MULTI_APPLIANCE
        self.NUM_SEQ_PER_BATCH= NUM_SEQ_PER_BATCH
        self.seq_length = seq_length
        self.series = series
        self.acc_threshold = acc_threshold
        self.num_appliance = len(self.appliances)
        self.acc_metric = defaultdict(lambda: 0)
        self.energy_metric = defaultdict(lambda: 0)

    def relative_error_in_total_energy(self):
        relative_energy = 0
        for target_idx in range(self.NUM_SEQ_PER_BATCH - 1):
            power_idx_start = target_idx * self.seq_length * self.num_appliance
            power_idx_end = (target_idx + 1) * self.seq_length * self.num_appliance

            target_series = self.target[power_idx_start:power_idx_end]
            pred_series = self.pred[power_idx_start:power_idx_end]
            for appliance_idx in range(self.num_appliance):
                appliance_energy_error = self.energy_calculation(target_series, pred_series, appliance_idx)
                label = self.appliances[appliance_idx]
                self.energy_metric[label] = self.energy_metric[label] + abs(appliance_energy_error)
                relative_energy = relative_energy + abs(appliance_energy_error)

        self.print_energy_metrics()
        return relative_energy/(self.NUM_SEQ_PER_BATCH*self.num_appliance)

    def energy_calculation(self, target_series, pred_series, appliance_idx):
        label_idx = [timeStamp * self.num_appliance + appliance_idx for timeStamp in range(self.seq_length)]
        sum_target = np.sum(target_series[label_idx])
        sum_pred = np.sum(pred_series[label_idx])
        relative_energy_error = (sum_pred - sum_target) / (max(sum_pred, sum_target) + 0.01)

        return relative_energy_error

    def print_energy_metrics(self):
        for label in self.appliances:
            print('Relative energy error of', label, '= {:.4f}'.format(self.energy_metric[label]/self.NUM_SEQ_PER_BATCH))

    def acc(self):
        result = 0
        for target_idx in range(self.NUM_SEQ_PER_BATCH):
            acc = self.average_accuracy_calculation(self.target, self.pred, target_idx)
            result = result + acc

        self.print_acc_metrics()
        return result/self.NUM_SEQ_PER_BATCH

    def average_accuracy_calculation(self, target_series, pred_series, target_idx):
        avg_acc = 0
        power_idx_start = target_idx * self.seq_length * self.num_appliance
        power_idx_end = (target_idx+1) * self.seq_length * self.num_appliance
        target = target_series[power_idx_start:power_idx_end]
        pred = pred_series[power_idx_start:power_idx_end]

        for appliance_idx in range(self.num_appliance):
            appliance_acc = self.accuracy_calculation(target, pred, appliance_idx)
            avg_acc = avg_acc + appliance_acc
            label = self.appliances[appliance_idx]
            self.acc_metric[label] = self.acc_metric[label] + appliance_acc

        return avg_acc/self.num_appliance

    def accuracy_calculation(self, target_series, pred_series, appliance_idx):
        accuracy = 0
        for timeStamp in range(self.seq_length):
            target_on = 0
            pred_on = 0
            idx = timeStamp - (self.series // 2)

            for i in range(self.series):
                label_idx = (idx + i) * self.num_appliance + appliance_idx

                if label_idx >=0 and label_idx < self.seq_length*self.num_appliance:
                    if target_series[label_idx] >= self.acc_threshold:
                        target_on = target_on + 1
                    if pred_series[label_idx] >= self.acc_threshold:
                        pred_on = pred_on + 1

            if (target_on - pred_on) < self.series/2:
                accuracy = accuracy + 1

        return accuracy/self.seq_length

    def print_acc_metrics(self):
        for label in self.appliances:
            print('On_Off accuracy of', label, '= {:.4f}%'.format(100*self.acc_metric[label]/self.NUM_SEQ_PER_BATCH))