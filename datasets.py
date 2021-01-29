import numpy as np
import pandas as pd
from sklearn import preprocessing
import math


def load_datasets_feature(filename):
    features_df = pd.read_csv(filename, delimiter='\\s*,\\s*', header=0)
    return features_df


def load_join_data3(features_df, result_file, histograms_path, num_rows, num_columns):
    cols = ['dataset1', 'dataset2', 'result_size', 'mbr_tests', 'duration']
    # Result DF contains dataset names, result cardinality, # of MBR tests, and duration in seconds
    result_df = pd.read_csv(result_file, delimiter='\\s*,\\s*', header=None, names=cols)
    # result_df = result_df.sample(frac=1)
    # Add dataset information of the first (left) dataset
    result_df = pd.merge(result_df, features_df, left_on='dataset1', right_on='dataset_name')
    # Add dataset information for the second (right) dataset
    result_df = pd.merge(result_df, features_df, left_on='dataset2', right_on='dataset_name')

    # Load histograms
    ds1_histograms, ds2_histograms, ds1_original_histograms, ds2_original_histograms, ds_all_histogram, ds_bops_histogram = load_histograms(
        result_df, histograms_path, num_rows, num_columns)

    #print(ds1_histograms.shape)
    #print(result_df.shape)
    #exit(0)

    # Compute BOPS
    # First, do an element-wise multiplication of the two histograms
    bops = np.multiply(ds1_original_histograms, ds2_original_histograms)
    # Reshape into a two dimensional array. First dimension represents the dataset number, e.g., first entry
    # represents the first dataset of each. Second dimension represents the values in the multiplied histograms
    bops = bops.reshape((bops.shape[0], num_rows * num_columns))
    # Sum the values in each row to compute the final BOPS value
    bops_values = np.sum(bops, axis=1)
    # The final reshape puts each BOPS value in an array with a single value. Thus it produces a 2D array.
    bops_values = bops_values.reshape((bops_values.shape[0], 1))
    result_df['bops'] = bops_values
    cardinality_x = result_df['cardinality_x']
    cardinality_y = result_df['cardinality_y']
    result_size = result_df['result_size']
    mbr_tests = result_df['mbr_tests']

    # Compute the join selectivity as result_cardinality/(cardinality x * cardinality y)
    result_df['join_selectivity'] = result_size / (cardinality_x * cardinality_y)

    # Compute the MBR selectivity in the same way
    result_df['mbr_tests_selectivity'] = mbr_tests / (cardinality_x * cardinality_y)

    # Apply MinMaxScaler to normalize numeric columns used in either training or testing to the range [0, 1]
    # The following transformation tries to adjust relevant columns to be scaled together

    column_groups = [
        ['duration'],
        ['AVG area_x', 'AVG area_y'],
        ['AVG x_x', 'AVG y_x', 'AVG x_y', 'AVG y_y'],
        ['E0_x', 'E2_x', 'E0_y', 'E2_y'],
        ['join_selectivity'],
        ['mbr_tests_selectivity'],
        ['cardinality_x', 'cardinality_y', 'result_size'],
        ['bops', 'mbr_tests']
    ]
    for column_group in column_groups:
        input_data = result_df[column_group].to_numpy()
        original_shape = input_data.shape
        reshaped = input_data.reshape(input_data.size, 1)
        reshaped = preprocessing.minmax_scale(reshaped)
        result_df[column_group] = reshaped.reshape(original_shape)
        #result_df[column_group] = scaler.fit_transform(result_df[column_group])

    return result_df, ds1_histograms, ds2_histograms, ds_all_histogram, ds_bops_histogram

def load_join_data(features_df, result_file, histograms_path, num_rows, num_columns):
    cols = ['dataset1', 'dataset2', 'result_size', 'mbr_tests', 'duration']
    # Result DF contains dataset names, result cardinality, # of MBR tests, and duration in seconds
    result_df = pd.read_csv(result_file, delimiter=',', header=None, names=cols)
    # result_df = result_df.sample(frac=1)
    # Add dataset information of the first (left) dataset
    result_df = pd.merge(result_df, features_df, left_on='dataset1', right_on='dataset_name')
    # Add dataset information for the second (right) dataset
    result_df = pd.merge(result_df, features_df, left_on='dataset2', right_on='dataset_name')

    # Load histograms
    ds1_histograms, ds2_histograms, ds1_original_histograms, ds2_original_histograms, ds_all_histogram, ds_bops_histogram = load_histograms(
        result_df, histograms_path, num_rows, num_columns)

    # Compute BOPS
    # First, do an element-wise multiplication of the two histograms
    bops = np.multiply(ds1_original_histograms, ds2_original_histograms)
    # Reshape into a two dimensional array. First dimension represents the dataset number, e.g., first entry
    # represents the first dataset of each. Second dimension represents the values in the multiplied histograms
    bops = bops.reshape((bops.shape[0], num_rows * num_columns))
    # Sum the values in each row to compute the final BOPS value
    bops_values = np.sum(bops, axis=1)
    # The final reshape puts each BOPS value in an array with a single value. Thus it produces a 2D array.
    bops_values = bops_values.reshape((bops_values.shape[0], 1))
    # result_df['bops'] = bops_values
    cardinality_x = result_df[' cardinality_x']
    cardinality_y = result_df[' cardinality_y']
    result_size = result_df['result_size']
    mbr_tests = result_df['mbr_tests']

    # Compute the join selectivity as result_cardinality/(cardinality x * cardinality y) * 10E+9
    join_selectivity = result_size / (cardinality_x * cardinality_y)
    join_selectivity = join_selectivity * 1E5

    # Compute the MBR selectivity in the same way
    mbr_tests_selectivity = mbr_tests / (cardinality_x * cardinality_y)
    mbr_tests_selectivity = mbr_tests_selectivity * 1E5

    duration = result_df['duration']

    dataset1 = result_df['dataset1']
    dataset2 = result_df['dataset2']

    # result_df = result_df.drop(columns=['result_size', 'dataset1', 'dataset2', 'dataset_name_x', 'dataset_name_y', ' cardinality_x', ' cardinality_y'])
    # result_df = result_df.drop(
    #     columns=['result_size', 'dataset1', 'dataset2', 'dataset_name_x', 'dataset_name_y'])

    result_df = result_df.drop(
        columns=['result_size', 'dataset1', 'dataset2', 'dataset_name_x', 'dataset_name_y', ' cardinality_x',
                 ' cardinality_y', 'mbr_tests', 'duration'])

    # Normalize all the values using MinMax scaler
    # These values are [AVG area_x, AVG x_x, AVG y_x, E0_x, E2_x, AVG area_y, AVG x_y, AVG y_y, E0_y, E2_y]
    x = result_df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    result_df = pd.DataFrame(x_scaled, columns=result_df.columns)

    result_df['cardinality_x'] = cardinality_x
    result_df['cardinality_y'] = cardinality_y
    result_df['bops'] = bops_values
    result_df['dataset1'] = dataset1
    result_df['dataset2'] = dataset2
    result_df.insert(len(result_df.columns), 'result_size', result_size, True)
    result_df.insert(len(result_df.columns), 'join_selectivity', join_selectivity, True)
    result_df.insert(len(result_df.columns), 'mbr_tests', mbr_tests, True)
    result_df.insert(len(result_df.columns), 'mbr_tests_selectivity', mbr_tests_selectivity, True)
    result_df.insert(len(result_df.columns), 'duration', duration, True)

    return result_df, ds1_histograms, ds2_histograms, ds_all_histogram, ds_bops_histogram


def load_join_data2(features_df, result_file, histograms_path, num_rows, num_columns):
    cols = ['count', 'dataset1', 'dataset2', 'result_size', 'mbr_tests', 'duration']
    result_df = pd.read_csv(result_file, delimiter=',', header=None, names=cols)

    # result_df = result_df.sample(frac=1)
    result_df = pd.merge(result_df, features_df, left_on='dataset1', right_on='dataset_name')
    result_df = pd.merge(result_df, features_df, left_on='dataset2', right_on='dataset_name')

    # Load histograms
    ds1_histograms, ds2_histograms, ds1_original_histograms, ds2_original_histograms, ds_all_histogram, ds_bops_histogram = load_histograms2(
        result_df, histograms_path, num_rows, num_columns)

    # Compute BOPS
    bops = np.multiply(ds1_original_histograms, ds2_original_histograms)
    # print (bops)
    bops = bops.reshape((bops.shape[0], num_rows * num_columns))
    bops_values = np.sum(bops, axis=1)
    bops_values = bops_values.reshape((bops_values.shape[0], 1))
    # result_df['bops'] = bops_values
    cardinality_x = result_df[' cardinality_x']
    cardinality_y = result_df[' cardinality_y']
    result_size = result_df['result_size']
    mbr_tests = result_df['mbr_tests']
    join_selectivity = result_size / (cardinality_x * cardinality_y)
    join_selectivity = join_selectivity * math.pow(10, 9)

    dataset1 = result_df['dataset1']
    dataset2 = result_df['dataset2']

    # result_df = result_df.drop(columns=['result_size', 'dataset1', 'dataset2', 'dataset_name_x', 'dataset_name_y', ' cardinality_x', ' cardinality_y'])
    # result_df = result_df.drop(
    #     columns=['result_size', 'dataset1', 'dataset2', 'dataset_name_x', 'dataset_name_y'])

    result_df = result_df.drop(
        columns=['count', 'result_size', 'dataset1', 'dataset2', 'dataset_name_x', 'dataset_name_y', ' cardinality_x',
                 ' cardinality_y', 'mbr_tests', 'duration'])

    x = result_df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    result_df = pd.DataFrame(x_scaled)

    result_df['cardinality_x'] = cardinality_x
    result_df['cardinality_y'] = cardinality_y
    result_df['bops'] = bops_values
    result_df['dataset1'] = dataset1
    result_df['dataset2'] = dataset2
    result_df.insert(len(result_df.columns), 'result_size', result_size, True)
    result_df.insert(len(result_df.columns), 'join_selectivity', join_selectivity, True)
    result_df.insert(len(result_df.columns), 'mbr_tests', join_selectivity, True)

    # print (len(result_df))
    # result_df.to_csv('result_df.csv')

    return result_df, ds1_histograms, ds2_histograms, ds_all_histogram, ds_bops_histogram


def load_histogram(histograms_path, num_rows, num_columns, dataset):
    hist = np.genfromtxt('{}/{}x{}/{}'.format(histograms_path, num_rows, num_columns, dataset), delimiter=',')
    normalized_hist = hist / hist.max()
    normalized_hist = normalized_hist.reshape((hist.shape[0], hist.shape[1], 1))
    hist = hist.reshape((hist.shape[0], hist.shape[1], 1))
    return normalized_hist, hist


def load_histogram2(histograms_path, num_rows, num_columns, count, dataset):
    hist = np.genfromtxt('{}/{}x{}/{}/{}'.format(histograms_path, num_rows, num_columns, count, dataset), delimiter=',')
    normalized_hist = hist / hist.max()
    normalized_hist = normalized_hist.reshape((hist.shape[0], hist.shape[1], 1))
    hist = hist.reshape((hist.shape[0], hist.shape[1], 1))
    return normalized_hist, hist


def load_histograms(result_df, histograms_path, num_rows, num_columns):
    ds1_histograms = []
    ds2_histograms = []
    ds1_original_histograms = []
    ds2_original_histograms = []
    ds_all_histogram = []
    ds_bops_histogram = []

    for dataset in result_df['dataset1']:
        normalized_hist, hist = load_histogram(histograms_path, num_rows, num_columns, dataset)
        ds1_histograms.append(normalized_hist)
        ds1_original_histograms.append(hist)

    for dataset in result_df['dataset2']:
        normalized_hist, hist = load_histogram(histograms_path, num_rows, num_columns, dataset)
        ds2_histograms.append(normalized_hist)
        ds2_original_histograms.append(hist)

    for i in range(len(ds1_histograms)):
        hist1 = ds1_original_histograms[i]
        hist2 = ds2_original_histograms[i]
        combined_hist = np.dstack((hist1, hist2))
        combined_hist = combined_hist / combined_hist.max()
        ds_all_histogram.append(combined_hist)

    for i in range(len(ds1_histograms)):
        hist1 = ds1_original_histograms[i]
        hist2 = ds2_original_histograms[i]
        bops_hist = np.multiply(hist1, hist2)
        if bops_hist.max() > 0:
            bops_hist = bops_hist / bops_hist.max()
        ds_bops_histogram.append(bops_hist)

    return np.array(ds1_histograms), np.array(ds2_histograms), np.array(ds1_original_histograms), np.array(
        ds2_original_histograms), np.array(ds_all_histogram), np.array(ds_bops_histogram)


def load_histograms2(result_df, histograms_path, num_rows, num_columns):
    ds1_histograms = []
    ds2_histograms = []
    ds1_original_histograms = []
    ds2_original_histograms = []
    ds_all_histogram = []
    ds_bops_histogram = []

    for index, row in result_df.iterrows():
        count = row['count']
        dataset1 = row['dataset1']
        dataset2 = row['dataset2']

        normalized_hist, hist = load_histogram2(histograms_path, num_rows, num_columns, count, dataset1)
        ds1_histograms.append(normalized_hist)
        ds1_original_histograms.append(hist)

        normalized_hist, hist = load_histogram2(histograms_path, num_rows, num_columns, count, dataset2)
        ds2_histograms.append(normalized_hist)
        ds2_original_histograms.append(hist)

    # count = 0
    # for dataset in result_df['dataset1']:
    #     count += 1
    #     normalized_hist, hist = load_histogram2(histograms_path, num_rows, num_columns, count, dataset)
    #     ds1_histograms.append(normalized_hist)
    #     ds1_original_histograms.append(hist)
    #
    # count = 0
    # for dataset in result_df['dataset2']:
    #     count += 1
    #     normalized_hist, hist = load_histogram2(histograms_path, num_rows, num_columns, count, dataset)
    #     ds2_histograms.append(normalized_hist)
    #     ds2_original_histograms.append(hist)

    for i in range(len(ds1_histograms)):
        hist1 = ds1_original_histograms[i]
        hist2 = ds2_original_histograms[i]
        combined_hist = np.dstack((hist1, hist2))
        combined_hist = combined_hist / combined_hist.max()
        ds_all_histogram.append(combined_hist)

    for i in range(len(ds1_histograms)):
        hist1 = ds1_original_histograms[i]
        hist2 = ds2_original_histograms[i]
        bops_hist = np.multiply(hist1, hist2)
        if bops_hist.max() > 0:
            bops_hist = bops_hist / bops_hist.max()
        ds_bops_histogram.append(bops_hist)

    return np.array(ds1_histograms), np.array(ds2_histograms), np.array(ds1_original_histograms), np.array(
        ds2_original_histograms), np.array(ds_all_histogram), np.array(ds_bops_histogram)


def main():
    print('Dataset utils')

    # features_df = load_datasets_feature('data/uniform_datasets_features.csv')
    # load_join_data(features_df, 'data/uniform_result_size.csv', 'data/histogram_uniform_values', 16, 16)

    features_df = load_datasets_feature('data/data_aligned/aligned_small_datasets_features.csv')
    join_data, ds1_histograms, ds2_histograms, ds_all_histogram = load_join_data(features_df,
                                                                                 'data/data_aligned/join_results_small_datasets.csv',
                                                                                 'data/data_aligned/histograms/small_datasets', 32,
                                                                                 32)
    print (join_data)


if __name__ == '__main__':
    main()
