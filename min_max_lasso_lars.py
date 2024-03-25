import pickle

import luigi
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, LassoLars


class LoadDiabetesData(luigi.Task):

    def output(self):
        return luigi.LocalTarget("diabetes.pkl")

    def run(self):
        diabetes = load_diabetes()
        df = pd.DataFrame(data=np.c_[diabetes['data'], diabetes['target']],
                          columns=diabetes['feature_names'] + ['target'])

        df.to_pickle(self.output().path)



class TrainTestSplit(luigi.Task):

    def output(self):
        return [
            luigi.LocalTarget("x_train.pkl"),
            luigi.LocalTarget("x_test.pkl"),
            luigi.LocalTarget("y_train.pkl"),
            luigi.LocalTarget("y_test.pkl")
        ]

    def requires(self):
        return LoadDiabetesData()

    def run(self):
        data = pd.read_pickle(self.input().path)
        X = data.drop(["target"], axis="columns")
        y = data[["target"]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        X_train.to_pickle(self.output()[0].path)
        X_test.to_pickle(self.output()[1].path)
        y_train.to_pickle(self.output()[2].path)
        y_test.to_pickle(self.output()[3].path)


class FitTransformMinMaxScaler(luigi.Task):

    def output(self):
        return [
            luigi.LocalTarget("minmax_scaled_x_train.pkl"),
            luigi.LocalTarget("minmax_scaled_x_test.pkl"),
            luigi.LocalTarget("minmax_scaler.pkl")
        ]

    def requires(self):
        return TrainTestSplit()

    def run(self):
        x_train = pd.read_pickle(self.input()[0].path)
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        scaled_x_train = pd.DataFrame(scaler.transform(x_train),
                                      columns=scaler.feature_names_in_,
                                      index=x_train.index)
        scaled_x_train.to_pickle(self.output()[0].path)

        x_test = pd.read_pickle(self.input()[1].path)
        scaler.transform(x_test)
        scaled_x_test = pd.DataFrame(scaler.transform(x_test),
                                     columns=scaler.feature_names_in_,
                                     index=x_test.index)
        scaled_x_test.to_pickle(self.output()[1].path)

        with open(self.output()[2].path, 'wb') as outfile:
            pickle.dump(scaler, outfile)


class TrainLassoLarsModelMinMaxScaled(luigi.Task):

    def output(self):
        return luigi.LocalTarget("lasso_lars_min_max.pkl")

    def requires(self):
        return [FitTransformMinMaxScaler(), TrainTestSplit()]

    def run(self):
        x_train = pd.read_pickle(self.input()[0][0].path)
        y_train = pd.read_pickle(self.input()[1][2].path)

        reg = LassoLars()
        reg.fit(x_train, y_train)

        with open(self.output().path, "wb") as outfile:
            pickle.dump(reg, outfile)


if __name__ == '__main__':
    luigi.build([TrainLassoLarsModelMinMaxScaled()], local_scheduler=True)