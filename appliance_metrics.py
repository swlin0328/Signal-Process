#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
from collections import defaultdict
import numpy as np

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
        self.single_acc_metric = defaultdict(lambda: 0)
        self.energy_metric = defaultdict(lambda: 0)
        self.mse_metric = defaultdict(lambda: 0)
        self.mabs_metric = defaultdict(lambda: 0)
        self.confusion_matrix = defaultdict(lambda: [0, 0, 0, 0])
        self.single_confusion_mat = defaultdict(lambda: [0, 0, 0, 0])

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
            appliance_acc, __ = self.accuracy_calculation(target, pred, appliance_idx)
            avg_acc = avg_acc + appliance_acc
            label = self.appliances[appliance_idx]
            self.acc_metric[label] = self.acc_metric[label] + appliance_acc

        return avg_acc/self.num_appliance

    def accuracy_calculation(self, target_series, pred_series, appliance_idx):
        accuracy = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
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
                if target_on > self.series/2:
                    TP = TP + 1
                else:
                    TN = TN + 1
            else:
                if pred_on > self.series/2:
                    FP = FP + 1
                else:
                    FN = FN + 1

        return accuracy/self.seq_length, [TP, TN, FP, FN]

    def print_acc_metrics(self):
        for label in self.appliances:
            print('On_Off accuracy of', label, '= {:.4f}%'.format(100*self.acc_metric[label]/self.NUM_SEQ_PER_BATCH))

    def cal_CF_mat(self, target_series, pred_series):
        for target_idx in range(self.NUM_SEQ_PER_BATCH):
            power_idx_start = target_idx * self.seq_length * self.num_appliance
            power_idx_end = (target_idx + 1) * self.seq_length * self.num_appliance
            target = target_series[power_idx_start:power_idx_end]
            pred = pred_series[power_idx_start:power_idx_end]

            for appliance_idx in range(self.num_appliance):
                __, confusion_matrix = self.accuracy_calculation(target, pred, appliance_idx)
                label = self.appliances[appliance_idx]
                self.confusion_matrix[label][0] = self.confusion_matrix[label][0] + confusion_matrix[0]
                self.confusion_matrix[label][1] = self.confusion_matrix[label][1] + confusion_matrix[1]
                self.confusion_matrix[label][2] = self.confusion_matrix[label][2] + confusion_matrix[2]
                self.confusion_matrix[label][3] = self.confusion_matrix[label][3] + confusion_matrix[3]

    def TPR_calculation(self):
        for appliance_idx in range(self.num_appliance):
            label = self.appliances[appliance_idx]
            TPR = self.confusion_matrix[label][0]/(self.confusion_matrix[label][0] + self.confusion_matrix[label][3] + 0.001)
            print('TPR of', label,
                  '= {:.4f}'.format(TPR))

    def FPR_calculation(self):
        for appliance_idx in range(self.num_appliance):
            label = self.appliances[appliance_idx]
            FPR = self.confusion_matrix[label][2]/(self.confusion_matrix[label][1] + self.confusion_matrix[label][2] + 0.001)
            print('FPR of', label,
                  '= {:.4f}'.format(FPR))

    def ROC(self):
        self.cal_CF_mat(self.target, self.pred)
        self.TPR_calculation()
        print('')
        self.FPR_calculation()
        print('')

    def MSE(self):
        self.MSE_calculation(self.target, self.pred)
        for appliance_idx in range(self.num_appliance):
            label = self.appliances[appliance_idx]
            print('MSE of', label, '= {:.4f}'.format(self.mse_metric[label]))
        print('')

    def MSE_calculation(self, target_series, pred_series):
        for target_idx in range(self.NUM_SEQ_PER_BATCH):
            power_idx_start = target_idx * self.seq_length * self.num_appliance
            power_idx_end = (target_idx + 1) * self.seq_length * self.num_appliance
            target = target_series[power_idx_start:power_idx_end]
            pred = pred_series[power_idx_start:power_idx_end]

            for appliance_idx in range(self.num_appliance):
                mse = self.appliance_mse(target, pred, appliance_idx)
                label = self.appliances[appliance_idx]
                self.mse_metric[label] = self.mse_metric[label] + mse

    def appliance_mse(self, target_series, pred_series, appliance_idx):
        mse = 0
        for timeStamp in range(self.seq_length):
            label_idx = timeStamp * self.num_appliance + appliance_idx
            mse = mse + (target_series[label_idx]-pred_series[label_idx])**2
        return mse/(self.seq_length*self.NUM_SEQ_PER_BATCH)

    def MABS(self):
        self.MABS_calculation(self.target, self.pred)
        for appliance_idx in range(self.num_appliance):
            label = self.appliances[appliance_idx]
            print('MABS of', label, '= {:.4f}'.format(self.mabs_metric[label]))
        print('')

    def MABS_calculation(self, target_series, pred_series):
        for target_idx in range(self.NUM_SEQ_PER_BATCH):
            power_idx_start = target_idx * self.seq_length * self.num_appliance
            power_idx_end = (target_idx + 1) * self.seq_length * self.num_appliance
            target = target_series[power_idx_start:power_idx_end]
            pred = pred_series[power_idx_start:power_idx_end]

            for appliance_idx in range(self.num_appliance):
                mabs = self.appliance_mabs(target, pred, appliance_idx)
                label = self.appliances[appliance_idx]
                self.mabs_metric[label] = self.mabs_metric[label] + mabs

    def appliance_mabs(self, target_series, pred_series, appliance_idx):
        mabs = 0
        for timeStamp in range(self.seq_length):
            label_idx = timeStamp * self.num_appliance + appliance_idx
            mabs = mabs + abs(target_series[label_idx]-pred_series[label_idx])
        return mabs/(self.seq_length*self.NUM_SEQ_PER_BATCH)

    def single_point_acc(self):
        for target_idx in range(self.NUM_SEQ_PER_BATCH):
            power_idx_start = target_idx * self.seq_length * self.num_appliance
            power_idx_end = (target_idx + 1) * self.seq_length * self.num_appliance
            target_series = self.target[power_idx_start:power_idx_end]
            pred_series = self.pred[power_idx_start:power_idx_end]

            for appliance_idx in range(self.num_appliance):
                acc, confusion_mat = self.single_acc_calculation(target_series, pred_series, appliance_idx)
                label = self.appliances[appliance_idx]
                self.single_acc_metric[label] = self.single_acc_metric[label] + acc
                self.single_confusion_mat[label][0] = self.single_confusion_mat[label][0] + confusion_mat[0]
                self.single_confusion_mat[label][1] = self.single_confusion_mat[label][1] + confusion_mat[1]
                self.single_confusion_mat[label][2] = self.single_confusion_mat[label][2] + confusion_mat[2]
                self.single_confusion_mat[label][3] = self.single_confusion_mat[label][3] + confusion_mat[3]

        self.print_single_point_metric()
        print('')
        self.print_single_point_TPR()
        print('')
        self.print_single_point_FPR()


    def single_acc_calculation(self, target_series, pred_series, appliance_idx):
        accuracy = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for idx in range(self.seq_length):
            label_idx = idx * self.num_appliance + appliance_idx

            if target_series[label_idx] >= self.acc_threshold and pred_series[label_idx] >= self.acc_threshold:
                accuracy =  accuracy + 1
                TP = TP + 1
            elif target_series[label_idx] >= self.acc_threshold and pred_series[label_idx] < self.acc_threshold:
                FN = FN + 1
            elif target_series[label_idx] < self.acc_threshold and pred_series[label_idx] < self.acc_threshold:
                accuracy =  accuracy + 1
                TN = TN + 1
            else:
                FP = FP + 1

        return accuracy/(self.NUM_SEQ_PER_BATCH*self.seq_length), [TP, TN, FP, FN]

    def print_single_point_metric(self):
        for appliance_idx in range(self.num_appliance):
            label = self.appliances[appliance_idx]
            print('Single point acc of', label, '= {:.4f}%'.format(self.single_acc_metric[label]*100))

    def print_single_point_TPR(self):
        for appliance_idx in range(self.num_appliance):
            label = self.appliances[appliance_idx]
            TPR = self.single_confusion_mat[label][0] / (
                        self.single_confusion_mat[label][0] + self.single_confusion_mat[label][3] + 0.001)
            print('TPR of', label,
                  '= {:.4f}'.format(TPR))

    def print_single_point_FPR(self):
        for appliance_idx in range(self.num_appliance):
            label = self.appliances[appliance_idx]
            FPR = self.single_confusion_mat[label][2]/(self.single_confusion_mat[label][1] + self.single_confusion_mat[label][2] + 0.001)
            print('FPR of', label,
                  '= {:.4f}'.format(FPR))

