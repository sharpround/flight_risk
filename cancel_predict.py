import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics
from os import listdir
import pickle
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

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
                   "CRSDepTime": np.float,
                   "DepTimeBlk": np.str,
                   "CRSArrTime": np.float,
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
                       "CRSArrTime",
                       "OrdinalDate",
                       "WeekDay",
                       "YearDay"]
    return continuous_vars


def get_feature_variables():
    feature_vars = ["CRSElapsedTime",
                    "Distance",
                    "CRSDepTime",
                    "CRSArrTime",
                    "UniqueCarrier",
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
                    "CancellationCode",
                    "WeekDay",
                    "YearDay"]
    return feature_vars


def get_target_column():
    pass



def read_all_csv(filenames, usecols=None, dtype_dict=None, verbose=False, frac=1.0):

    if type(filenames) is not list:
        print("function read_all_csv requires the input to be a list of filenames")
        return None # TODO: return appropriate exception/error

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
    # df["WeekDay"] = df["WeekDay"] / 6.0
    return df



def make_day_of_year_feature(df, column_name):
    df["YearDay"] = df[column_name].apply(lambda x: x.timetuple().tm_yday)
    # df["YearDay"] = df["YearDay"] / 365.25
    return df



def make_ordinal_date_feature(df, column_name):
    df["OrdinalDate"] = df[column_name].apply(lambda x: x.toordinal())
    return df



def transform_date_feature(df, column_name, format="%Y-%m-%d"):
    df[column_name] = pd.to_datetime(df[column_name], format=format)
    return df



def normalize_continuous_feature(df, column_name):
    meanval = df[column_name].mean()
    stdval = df[column_name].std()
    df[column_name] = (df[column_name] - meanval) / stdval
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


def contiguous_train_test_split(X, y, train_size):

    N = len(y)
    N_train = np.int(N*train_size)

    start_train = np.random.randint(N)
    end_train = start_train + N_train


    if end_train <= N:
        print("start: " + str(start_train))
        print("end: " + str(end_train))
        print("total: " + str(N))
        X_train = X[start_train:end_train, :]
        y_train = y[start_train:end_train]
        
        X1 = X[:start_train, :]
        X2 = X[end_train:, :]
        X_test = sparse.vstack((X1, X2))
        y_test = np.r_[y[:start_train], y[end_train:]]

    else:
        end_train = end_train - N
        print("start: " + str(start_train))
        print("end: " + str(end_train))
        print("total: " + str(N))
        X1 = X[:end_train, :]
        X2 = X[start_train:, :]
        X_train = sparse.vstack((X1, X2))
        y_train = np.r_[y[:end_train], y[start_train:]]

        X_test = X[end_train:start_train, :]
        y_test = y[end_train:start_train]

    return X_train, X_test, y_train, y_test


def preprocess_ontime_data(verbose=True):
    ### load data

    filenames = get_filenames(ONTIME_PATH)
    usecols = get_data_columns()

    # filenames = ["On_Time_On_Time_Performance_2012_7.csv"]

    if verbose:
        print("files to load: " + str(filenames))
        print("columns to import: " + str(usecols))

    df = read_all_csv(filenames,
                      usecols=usecols,
                      dtype_dict=get_dtype_dict(),
                      verbose=verbose,
                      frac=0.01
                      )
    if verbose:
        print("Size of data set: " + str(len(df)))

    ### encode categorical features

    categorical_vars = get_categorical_vars()

    transformer_dict = {}
    for var in categorical_vars:
        df, transformer = transform_categorical_feature(df, var)
        transformer_dict[var] = transformer

    if verbose:
        print("transformer: " + str(transformer_dict))

    ### make new features

    df = transform_date_feature(df, "FlightDate")

    df = make_day_of_week_feature(df, "FlightDate")

    df = make_day_of_year_feature(df, "FlightDate")

    df = make_ordinal_date_feature(df, "FlightDate")

    ### normalize features to mean = 0, std = 1

    continuous_vars = get_continuous_vars()
    for var in continuous_vars:
        df = normalize_continuous_feature(df, var)

    return df, transformer_dict


def plot_learning_curve(clf, X_train, X_test, y_train, y_test):
    pass


def vectorize_data(df):

    cat_vars = ["UniqueCarrier",
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
                "DestState"]

    con_vars = ["CRSElapsedTime",
                "Distance",
                "CRSDepTime",
                "CRSArrTime",
                "WeekDay",
                "YearDay"]

    df = df.dropna()

    Xenc = OneHotEncoder()

    X1 = df[con_vars].as_matrix()
    X2 = Xenc.fit_transform(df[cat_vars].as_matrix())

    X = sparse.hstack((X1, X2))
    X = X.tocsr()

    y = df["Cancelled"].as_matrix()

    return X, y



def main():
    # df, transformer_dict = preprocess_ontime_data()
    #
    # print("saving dataset to disk")
    # pickle.dump(df, open("ontime_sample_01.pickle", 'wb'))
    # pickle.dump(transformer_dict, open("transformer_dict_01.pickle", 'wb'))

    df = pickle.load(open("ontime_sample_01.pickle", 'rb'))
    transformer_dict = pickle.load(open("transformer_dict_01.pickle", 'rb'))

    df = df.sort_values(by="OrdinalDate")

    X, y = vectorize_data(df)

    print("X shape: " + str(X.shape))
    print("y shape: " + str(y.shape))

    clf = linear_model.LogisticRegression(penalty='l2', verbose=True)

    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, train_size=0.7)
    X_train, X_test, y_train, y_test = contiguous_train_test_split(X, y, train_size=0.7)

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)

    y_baseline = np.ones(y_test.shape) * (np.sum(y) / len(y))

    clf_score       = metrics.log_loss(y_test, y_pred)
    baseline_score  = metrics.log_loss(y_test, y_baseline)
    never_score     = metrics.log_loss(y_test, np.zeros(y_test.shape))
    always_score    = metrics.log_loss(y_test, np.ones(y_test.shape))

    print("log-loss score of classifier: "  + str(clf_score))
    print("log-loss score of baseline: "    + str(baseline_score))
    print("log-loss score of never: "       + str(never_score))
    print("log-loss score of always: "      + str(always_score))

    clf_score       = metrics.brier_score_loss(y_test, y_pred[:, 1])
    baseline_score  = metrics.brier_score_loss(y_test, y_baseline)
    never_score     = metrics.brier_score_loss(y_test, np.zeros(y_test.shape))
    always_score    = metrics.brier_score_loss(y_test, np.ones(y_test.shape))

    print("Brier loss of classifier: "  + str(clf_score))
    print("Brier loss of baseline: "    + str(baseline_score))
    print("Brier loss of never: "       + str(never_score))
    print("Brier loss of always: "      + str(always_score))

    clf_score       = metrics.roc_auc_score(y_test, y_pred[:, 1])
    baseline_score  = metrics.roc_auc_score(y_test, y_baseline)
    never_score     = metrics.roc_auc_score(y_test, np.zeros(y_test.shape))
    always_score    = metrics.roc_auc_score(y_test, np.ones(y_test.shape))

    print("ROC AUC of classifier: "  + str(clf_score))
    print("ROC AUC score of baseline: "    + str(baseline_score))
    print("ROC AUC score of never: "       + str(never_score))
    print("ROC AUC score of always: "      + str(always_score))


    print("program complete")



if __name__ == '__main__':
    main()
