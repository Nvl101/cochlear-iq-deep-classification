'''
model evaluation methods
- calculate accuracy
'''
from abc import abstractclassmethod
from typing import Any, Iterable
from dataio.utility import denormalize_label
import torch
from torch.utils.data import DataLoader


class Evaluator:
    '''evaluator abstract class'''
    @abstractclassmethod
    def append(self, *values):
        pass

    @abstractclassmethod
    def __str__(self) -> str:
        pass


class LabelsEvaluator:
    labels_pred_raw: list = []
    labels_actual_raw: list = []

    def append(self, labels_pred: torch.Tensor, labels_actual: torch.Tensor):
        '''
        append prediction labels and actual labels
        '''
        assert len(self.labels_pred_raw) == len(self.labels_actual_raw), \
            'pred and actual label lengths not equal'
        assert len(labels_pred) == len(labels_actual)
        self.labels_pred_raw.extend(
            [label for label in torch.unbind(labels_pred)])
        self.labels_actual_raw.extend(
            [label for label in torch.unbind(labels_actual)])


class AccuracyEvaluator(LabelsEvaluator):
    '''
    calculate classification accuracy

    example usage:
    accuracy_eval = AccuracyCalcula
    for images, labels in some_dataset:
        labels_pred = model(images)
        accuracy_eval.append(labels_pred, labels)
    accuracy_eval.value()
    '''
    _denormalize_method: callable

    def __init__(
            self,
            labels_pred: list = None,
            labels_actual: list = None,
            ):
        '''
        if provided following arguments, it will set predicted and actual labels
        arguments:
            labels_pred: predicted labels
            labels_actual: actual labels
        '''
        self._denormalize_method = denormalize_label
        if labels_pred and labels_actual:
            assert(len(labels_pred) == len(labels_actual))
            self.labels_pred_raw = [label for label in labels_pred]
            self.labels_actual_raw = [label for label in labels_actual]

    @property
    def labels_pred(self):
        return [self._denormalize_method(lbl) for lbl in self.labels_pred_raw]

    @property
    def labels_actual(self):
        return [self._denormalize_method(lbl) for lbl in self.labels_actual_raw]

    def accuracy(self):
        ''' calculate overall accuracy'''
        true_preds = 0
        total_preds = len(self.labels_pred)
        for label_pred, label_raw in zip(self.labels_pred, self.labels_actual):
            if label_pred ==  label_raw:
                true_preds += 1
        val_accuracy = float(true_preds) / total_preds
        return val_accuracy

    def reset_labels(self):
        '''
        reset labels at the start of each epoch
        so that the accuracy, precision and recalls do not accumulate
        '''
        self.labels_actual_raw = []
        self.labels_pred_raw = []

    def _tfpn(self, true_values: Iterable[int] = [3]):
        '''
        calculates true positive, false positive, true negative, false negative
        '''
        is_true = lambda x: x in true_values
        tp, fp, tn, fn = 0, 0, 0, 0
        for label_pred, label_actual in zip(self.labels_pred, self.labels_actual):
            if is_true(label_pred):
                if is_true(label_actual):
                    tp += 1
                else:
                    fp += 1
            else:
                if is_true(label_actual):
                    fn += 1
                else:
                    tn += 1
        return tp, fp, tn, fn

    def precision(self, true_values: Iterable[int] = [3]):
        ''' calculate precision value'''
        tp, fp, _, _ = self._tfpn(true_values)
        try:
            precision_val = tp / (tp + fp)
        except ZeroDivisionError:
            precision_val = -1
        return precision_val

    def recall(self, true_values: Iterable[int] = [3]):
        ''' calculate recall value '''
        tp, _, _, fn = self._tfpn(true_values)
        try:
            recall_val = tp / (tp + fn)
        except ZeroDivisionError:
            recall_val = -1
        return recall_val

    def f1(self, true_values: Iterable[int] = [3]):
        ''' calculate f1 score'''
        precision = self.precision(true_values)
        recall = self.recall(true_values)
        f1_val = 2 * (precision * recall) / (precision + recall)
        return f1_val

    # def accuracy(self):
    #     '''
    #     calculate value of accuracy
    #     '''
    #     correct_labels = 0
    #     total_labels = len(self.labels_pred_raw)
    #     if total_labels == 0:
    #         return 0
    #     for i in range(len(self.labels_pred_raw)):
    #         label_pred = self.labels_pred_raw[i]
    #         label_actual = self.labels_actual_raw[i]
    #         if self._denormalize_method(label_pred) \
    #             == self._denormalize_method(label_actual):
    #             correct_labels += 1
    #     return float(correct_labels) / total_labels

    def __str__(self):
        # percent = lambda x: round(x * 100, 2)
        return f'{round(self.accuracy() * 100, 2)} %'
        # return f'Evaluator of accuracy {percent(accuracy_percent)} %  \
        #     precision {round(self.precision(), 2)}  \
        #         recall {round(self.recall(), 2)}  \
        #             f1 {round(self.f1(), 2)}'

class LossEvaluator(LabelsEvaluator):
    loss_function: callable
    losses: list = []  # alternatively calculate the average of the losses

    def __init__(self, criterion, labels_pred=None, labels_actual=None):
        self.loss_function = criterion
        self.losses = []
        if labels_pred and labels_actual:
            assert len(labels_pred) == len(labels_actual), \
                'pred and actual label lengths not equal'
            self.labels_pred_raw = labels_pred
            self.labels_actual_raw = labels_actual

    def append_loss(self, loss):
        '''
        append the loss value into  self.losses
        this method should be depreciated when __call__ is tested
        '''
        self.losses.append(loss)

    def value(self):
        '''
        calculate loss value
        '''
        if len(self.losses) > 0:
            return float(sum(self.losses) / len(self.losses))
        else:
            return self.loss_function(torch.stack(self.labels_pred_raw), torch.stack(self.labels_actual_raw))

    def __str__(self):
        loss_value = round(self.value(), 3)
        return str(loss_value)

    def __call__(self, *args, **kwargs):
        loss_value = self.loss_function(*args, **kwargs)
        self.append_loss(loss_value.item())
        return loss_value


# debugging
if __name__ == '__main__':
    pass
