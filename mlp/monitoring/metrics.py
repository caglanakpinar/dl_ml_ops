import numpy as np
from keras import metrics

from mlp import log


class Metrics:
    @staticmethod
    def residuals(actual, predicted):
        """residuals for MOV Adj. Score - Actual Mov Adj Score
        residual = abs(MOV Adj. Score - Actual Mov Adj Score)
        """
        residuals = actual - predicted
        log(log.info, f" *** mean predicted:  {np.mean(predicted)}")
        log(log.info, f" *** std predicted:  {np.std(predicted)}")
        log(log.info, f" *** max predicted:  {np.max(predicted)}")
        log(log.info, f" *** min predicted:  {np.min(predicted)}")
        log(log.info, f" *** mean residuals:  {np.mean(residuals)}")
        log(log.info, f" *** std residuals: {np.std(residuals)}")
        log(log.info, f" *** max residuals: {np.max(residuals)}")
        log(log.info, f" *** min residuals: {np.min(residuals)}")

    @staticmethod
    def train_epoch_metrics(name):
        return {
            "tp": metrics.TruePositives(name="tp", thresholds=0.01),
            "fp": metrics.FalsePositives(name="fp", thresholds=0.01),
            "tn": metrics.TrueNegatives(name="tn", thresholds=0.01),
            "fn": metrics.FalseNegatives(name="fn", thresholds=0.01),
            "accuracy": metrics.BinaryAccuracy(name="accuracy"),
            "precision": metrics.Precision(
                name="precision",
            ),
            "recall": metrics.Recall(name="recall"),
            "auc": metrics.AUC(name="auc"),
            "prc": metrics.AUC(name="prc", curve="PR"),
        }.get(name, name)
