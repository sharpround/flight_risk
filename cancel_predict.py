import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics



def get_filenames():
	filenames = "on_time/On_Time_On_Time_Performance_2015_9.csv"
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



def load_data(filenames, usecols=None):

	df = pd.DataFrame()
	df.append(pd.read_csv(filename, usecols=usecols))
	
	return df



def clean_data(df):
	pass



def transform_categorical_feature(df, column_name):
    unique_values = df[column_name].unique()
    transformer_dict = {}
    for ii, value in enumerate( unique_values ):
        transformer_dict[value] = ii

    def label_map(y):
        return transformer_dict[y]
    
    df[column_name] = df[column_name].apply( label_map )
    return df, transformer_dict



def transform_time_feature(df, column_name):
	pass


def normalize_continuous_feature(df, column_name):
	pass



def compute_score(y, y_pred):
	score = metrics.log_loss(y, y_pred)
	return score



def main():
	filename = get_filenames()
	usecols = get_data_columns()

	pass



if __name__ == '__main__':
	main()