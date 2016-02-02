import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics
from os import listdir

data_path = "D:/data_science/freebird/on_time"


def get_filenames(path):
    filenames = listdir(path)
    return filenames


def get_data_columns():
    columns = ["FlightDate",
               "UniqueCarrier",
               "OriginAirportID",
               "OriginAirportSeqID",
               "OriginCityMarketID",
               "OriginState",
               "DestAirportID",
               "DestAirportSeqID",
               "DestCityMarketID",
               "DestState",
               "CRSDepTime",
               "DepTimeBlk",
               "CRSArrTime",
               "ArrTimeBlk",
               "CRSElapsedTime",
               "Distance",
               "DistanceGroup",
               "Cancelled",
               "CancellationCode"]
    return columns


def read_all_csv(filenames, usecols=None):
    l = []
    for f in filenames:
        tmp_df = pd.read_csv(f, index_col=None, header=0)
        l.append(tmp_df)
    df = pd.concat(l)
    return df


def clean_data(df):
    pass


def transform_categorical_feature(df, column_name):
    unique_values = df[column_name].unique()
    transformer_dict = {}
    for ii, value in enumerate(unique_values):
        transformer_dict[value] = ii

    def label_map(y):
        return transformer_dict[y]

    df[column_name] = df[column_name].apply(label_map)
    return df, transformer_dict


def transform_time_feature(df, column_name):
    pass


def normalize_continuous_feature(df, column_name):
    pass


def compute_score(y, y_pred):
    score = metrics.log_loss(y, y_pred)
    return score


def main():
    filenames = get_filenames(data_path)
    usecols = get_data_columns()

    print(get_filenames())
    print(usecols)


if __name__ == '__main__':
    main()
