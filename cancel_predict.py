import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics
from os import listdir

ONTIME_PATH = "on_time/"
TICKET_PATH = "bts-db1b/DB1B/"


def get_filenames(path):
    filenames = listdir(path)
    return filenames


def get_dtype_dict():
    dtype_dict = {"FlightDate": np.str,
                   "UniqueCarrier": np.str,
                   "OriginAirportID": np.uint,
                   "OriginAirportSeqID": np.uint,
                   "OriginCityMarketID": np.uint,
                   "OriginState": np.str,
                   "DestAirportID": np.uint,
                   "DestAirportSeqID": np.uint,
                   "DestCityMarketID": np.uint,
                   "DestState": np.str,
                   "CRSDepTime": np.uint,
                   "DepTimeBlk": np.str,
                   "CRSArrTime": np.uint,
                   "ArrTimeBlk": np.str,
                   "CRSElapsedTime": np.float,
                   "Distance": np.float,
                   "DistanceGroup": np.uint,
                   "Cancelled": np.bool,
                   "CancellationCode": np.str}
    return dtype_dict


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


def get_categorical_vars():
    categorical_vars = ["UniqueCarrier",
                        "OriginAirportID",
                        "OriginAirportSeqID",
                        "OriginCityMarketID",
                        "OriginState",
                        "DestAirportID",
                        "DestAirportSeqID",
                        "DestCityMarketID",
                        "DepTimeBlk",
                        "ArrTimeBlk",
                        "DistanceGroup",
                        "DestState",
                        "Cancelled",
                        "CancellationCode"]
    return categorical_vars



def get_continuous_vars():
    continuous_vars = ["CRSElapsedTime",
                       "Distance",
                       "CRSDepTime",
                       "CRSArrTime"]
    return continuous_vars



def read_all_csv(filenames, usecols=None, dtype_dict=None, verbose=False, frac=1.0):

    if type(filenames) is not list:
        print("function read_all_csv requires the input to be a list of filenames")
        return None # TODO: return appropriate error message

    df = pd.DataFrame()
    for fname in filenames:
        if verbose:
            print("loading " + fname)
        tmp_df = pd.read_csv(ONTIME_PATH + fname,
                             usecols=usecols,
                             header=0,
                             dtype=dtype_dict
                             )

        df = df.append(tmp_df.sample(frac=frac))

    return df



def transform_categorical_feature(df, column_name):
    unique_values = df[column_name].unique()
    transformer_dict = {}
    for ii, value in enumerate(unique_values):
        transformer_dict[value] = ii

    df[column_name] = df[column_name].apply(lambda x: transformer_dict[x])

    return df, transformer_dict
    # TODO: save transformer_dict to JSON so it can be used for consistently on all data


def make_day_of_week_feature(df, column_name):
    df["WeekDay"] = df[column_name].apply(lambda x: x.timetuple().tm_wday)
    df["WeekDay"] = df["WeekDay"] / 6.0
    return df



def make_day_of_year_feature(df, column_name):
    df["YearDay"] = df[column_name].apply(lambda x: x.timetuple().tm_yday)
    df["YearDay"] = df["YearDay"] / 365.25
    return df



def transform_date_feature(df, column_name, format="%Y-%m-%d"):
    df[column_name] = pd.to_datetime(df[column_name], format=format)
    return df



def normalize_continuous_feature(df, column_name, from_range=None, to_range=(0, 1)):
    minval = df[column_name].min()
    maxval = df[column_name].max()
    df[column_name] = (df[column_name] - minval) / (maxval - minval)
    return df



def compute_score(y, y_pred, scorer='log_loss'):
    if scorer=='log_loss':
        score = metrics.log_loss(y, y_pred)
    else:
        score = np.NaN
        # TODO: `maybe want to throw an appropriate error or warning
    return score



def record_to_features(filename):
    # take a filename, return a sparse array of features and titles for those features
    usecols = get_data_columns()
    df = pd.read_csv(filename, usecols=usecols, header=0)

    return X, X_headers



def main():
    ### load data

    filenames = get_filenames(ONTIME_PATH)
    usecols = get_data_columns()

    filenames = filenames[48:60]

    print("files to load: " + str(filenames))
    print("columns to import: " + str(usecols))

    # test_file = "on_time/On_Time_On_Time_Performance_2013_11.csv"

    df = read_all_csv(filenames,
                      usecols=usecols,
                      dtype_dict=get_dtype_dict(),
                      verbose=True,
                      frac=0.1
                      )


    ### encode categorical features

    categorical_vars = get_categorical_vars()

    transformer_dict = {}
    for var in categorical_vars:
        df, transformer = transform_categorical_feature(df, var)
        transformer_dict[var] = transformer

    print(df.head())
    print("transformer: " + str(transformer_dict))

    ### make new features

    df = transform_date_feature(df, "FlightDate")

    df = make_day_of_week_feature(df, "FlightDate")

    df = make_day_of_year_feature(df, "FlightDate")

    ### normalize features

    continuous_vars = get_continuous_vars()
    for var in continuous_vars:
        df = normalize_continuous_feature(df, var)

    print(df.head())


if __name__ == '__main__':
    main()
