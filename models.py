from keras.models import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import concatenate
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam, Nadam, RMSprop
from keras.losses import logcosh, binary_crossentropy, mean_absolute_percentage_error, mean_absolute_error, cosine_similarity, mean_squared_logarithmic_error
from keras.activations import relu, elu, sigmoid
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError, MeanAbsolutePercentageError
import talos as ta

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import datasets


def create_mlp(dim, regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(4, input_dim=dim, activation="relu"))
    model.add(Dense(2, activation="relu"))

    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))

    # return our model
    return model


def create_cnn(width, height, depth, parameters, filters=(4, 8, 16), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    input_shape = (height, width, depth)
    chan_dim = -1

    # define the model input
    inputs = Input(shape=input_shape)

    conv_size = parameters["cnn_conv_size"]
    pool_size = parameters["cnn_pool_size"]
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs

        # CONV => RELU => BN => POOL
        x = Conv2D(f, (conv_size, conv_size), padding="same")(x)
        x = Activation(parameters['cnn_activation'])(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(parameters["cnn_neurons"])(x)
    x = Activation(parameters['cnn_activation'])(x)
    x = BatchNormalization(axis=chan_dim)(x)
    x = Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(parameters["cnn_last_neurons"])(x)
    x = Activation(parameters['cnn_activation'])(x)

    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation=parameters['cnn_last_activation'])(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model

def run2(tabular_path, histogram_path, join_result_path, model_path, model_weights_path, is_train=True):
    print('Training the join cardinality estimator')
    print('Tabular data: {}'.format(tabular_path))
    print('Histogram path: {}'.format(histogram_path))
    print('Join result data: {}'.format(join_result_path))

    num_rows, num_columns = 16, 16
    # A list of datasets and their statistics
    # dataset_name,cardinality,avg_area,avg_x,avg_y,E0,E2
    tabular_features_df = datasets.load_datasets_feature(tabular_path)
    # Keep only uniform data
    #tabular_features_df = tabular_features_df[tabular_features_df.dataset_name.str.contains('niform')]

    # Load all the information needed to train a model for the above datasets.
    # This includes, the join results and the histograms
    # join_data: Contains the following features for each dataset [AVG area, AVG x, AVG y, E0, E2]
    # and the common attributes [bops, result_size, join_selectivity, mbr_tests, mbr_tests_selectivity, duration]
    # And some other (mostly unused attributes)
    join_data, ds1_histograms, ds2_histograms, ds_all_histogram, ds_bops_histogram = datasets.load_join_data3(
        tabular_features_df, join_result_path, histogram_path, num_rows, num_columns)
    join_data.to_csv("train_data.csv")

    print(f"Training and testing on {join_data.size} records")
    feature_cols = ['AVG area_x', 'AVG x_x', 'AVG y_x', 'E0_x', 'E2_x', 'AVG area_y',
       'AVG x_y', 'AVG y_y', 'E0_y', 'E2_y', 'cardinality_x', 'cardinality_y', 'bops']
    #feature_cols = ['AVG area_x', 'AVG area_y']
    target = 'join_selectivity'
    #target = 'result_size'
    #target = 'mbr_tests_selectivity'

    x_all = join_data[feature_cols].to_numpy()
    h_all = ds_all_histogram
    y_all = join_data[target].to_numpy()
    x_train_valid, x_test, h_train_valid, h_test, y_train_valid, y_test = train_test_split(x_all, h_all, y_all, test_size=0.20, random_state=42)
    x_train, x_valid, h_train, h_valid, y_train, y_valid = train_test_split(x_train_valid, h_train_valid, y_train_valid, test_size=0.20, random_state=42)

    p = {'lr': [0.01, 0.1, 0.001],
         'num_mlp_features': [len(feature_cols)],
         'first_neuron': [2, 4, 8],
         'hidden_layers': [2, 4, 8, 12],
         'hidden_neurons': [4, 8, 12],
         'batch_size': [256],
         'epochs': [300],
         #'optimizer': [Adam, Nadam, RMSprop],
         'optimizer': [Adam],
         'loss': [mean_absolute_error],
         'activation': [sigmoid, relu, elu],
         'last_activation': [sigmoid, relu, elu]}
    use_histogram = True
    if use_histogram:
        p['histogram_size'] = [h_all.shape[1]]
        p['cnn_depth'] = [h_all.shape[3]]
        p['cnn_activation'] = [relu, elu, sigmoid]
        p['cnn_conv_size'] = [3, 5, 7]
        p['cnn_pool_size'] = [2]
        p['cnn_neurons'] = [4, 8, 12]
        p['cnn_last_neurons'] = [4, 8]
        p['cnn_last_activation'] = [relu, elu, sigmoid]
        train_data = [x_train, h_train]
        valid_data = [x_valid, h_valid]
    else:
        train_data = x_train
        valid_data = x_valid

    p['loss'] = [mean_absolute_error]
    p['activation'] = [elu]
    p['batch_size'] = [256]
    p['epochs'] = [300]
    p['first_neuron'] = [8]
    p['hidden_layers'] = [4]
    p['hidden_neurons'] = [4]
    p['last_activation'] = [elu]
    p['loss'] = [mean_absolute_error]
    p['lr'] = [0.001]
    p['num_mlp_features'] = [13]

    t = ta.Scan(x=train_data, y=y_train, model=build_model,
                x_val=valid_data, y_val=y_valid,
                params=p, experiment_name='uniform',
                fraction_limit=0.1
                )
    best_model = t.best_model(metric='val_loss', asc=True)
    # Print out the parameters of the best model to be able to reproduce the best results easily
    best_model_parameters = t.data.nsmallest(n=1, columns='val_loss')
    for column in best_model_parameters.columns:
        print("p['{}'] = [{}]".format(column, best_model_parameters[column].iloc[0]))
    #print(t.data.nsmallest(n=1, columns='val_loss'))

    print('Testing')

    #models = t.evaluate_models(x_val=x_test, y_val=y_test, n_models=10, metric='loss', folds=5, asc=True)
    #print(models)

    if use_histogram:
        y_pred = best_model.predict([x_test, h_test])
    else:
        y_pred = best_model.predict(x_test)
    print(np.vstack((y_test, y_pred.flatten())).T)

    print('r2 score: {}'.format(r2_score(y_test, y_pred)))
    mse = MeanSquaredError()
    print(f'MSE {mse(y_test, y_pred).numpy()}')
    mape = MeanAbsolutePercentageError()
    print(f'MAPE {mape(y_test, y_pred).numpy()}')
    print(f'Mean Square Logarithmic Error {np.mean(mean_squared_logarithmic_error(y_test, y_pred))}')
    #print(f'Cosine Similarity {cosine_similarity(y_test.double(), y_pred.double())}')

    # diff = y_pred.flatten() - y_test
    # percent_diff = (diff / y_test)
    # abs_percent_diff = np.abs(percent_diff)
    #
    # # Compute the mean and standard deviation of the absolute percentage difference
    # mean = np.mean(abs_percent_diff)
    # std = np.std(abs_percent_diff)
    #
    # # NOTICE: mean is the MAPE value, which is the target we want to minimize
    # print('mean = {}, std = {}'.format(mean, std))


def build_model(x_train, y_train, x_val, y_val, params):
    mlp_model = Sequential()
    num_mlp_features = params['num_mlp_features']
    mlp_model.add(Dense(params['first_neuron'], input_dim=num_mlp_features, activation=params['activation']))
    for x in range(params['hidden_layers']):
        mlp_model.add(Dense(params['hidden_neurons'], activation=params['activation']))

    mlp_model.add(Dense(1, activation=params['last_activation']))

    # Create the CNN model for the histogram part
    if 'histogram_size' in params and params['histogram_size'] > 0:
        histogram_size = params['histogram_size']
        cnn_model = create_cnn(histogram_size, histogram_size, params['cnn_depth'], params)

        # Combine the MLP and CNN models
        x = Dense(4, activation=params['activation'])(concatenate([mlp_model.output, cnn_model.output]))
        x = Dense(1, activation=params['last_activation'])(x)

        # model = Model(inputs=[mlp.input, cnn1.input, cnn2.input, cnn3.input], outputs=x)
        model = Model(inputs=[mlp_model.input, cnn_model.input], outputs=x)
    else:
        model = mlp_model

    # Initialize the optimizer
    opt = params['optimizer'](lr=params['lr'])

    model.compile(loss=params['loss'], optimizer=opt, metrics=[mean_absolute_percentage_error, mean_absolute_error])

    es = EarlyStopping(monitor='val_loss', mode='min')

    history = model.fit(x=x_train, y=y_train,
                        validation_data=(x_val, y_val),
                        batch_size=params['batch_size'],
                        callbacks=[es],
                        epochs=params['epochs'], verbose=0)

    return history, model


def run(tabular_path, histogram_path, join_result_path, model_path, model_weights_path, is_train=True):
    print ('Training the join cardinality estimator')
    print ('Tabular data: {}'.format(tabular_path))
    print ('Histogram path: {}'.format(histogram_path))
    print ('Join result data: {}'.format(join_result_path))

    target = 'join_selectivity'
    num_rows, num_columns = 16, 16
    # A list of datasets and their statistics
    # dataset_name,cardinality,avg_area,avg_x,avg_y,E0,E2
    tabular_features_df = datasets.load_datasets_feature(tabular_path)

    # Load all the information needed to train a model for the above datasets.
    # This includes, the join results and the histograms
    # join_data: Contains the following features for each dataset [AVG area, AVG x, AVG y, E0, E2]
    # and the common attributes [bops, result_size, join_selectivity, mbr_tests, mbr_tests_selectivity, duration]
    # And some other (mostly unused attributes)
    join_data, ds1_histograms, ds2_histograms, ds_all_histogram, ds_bops_histogram = datasets.load_join_data(
        tabular_features_df, join_result_path, histogram_path, num_rows, num_columns)

    num_features = len(join_data.columns) - 10

    if is_train:
        train_attributes, test_attributes, ds1_histograms_train, ds1_histograms_test, ds2_histograms_train, ds2_histograms_test, ds_all_histogram_train, ds_all_histogram_test, ds_bops_histogram_train, ds_bops_histogram_test = train_test_split(
            join_data, ds1_histograms, ds2_histograms, ds_all_histogram, ds_bops_histogram, test_size=0.20,
            random_state=42)
        X_train = pd.DataFrame.to_numpy(train_attributes[join_data.columns[:num_features]])
        X_test = pd.DataFrame.to_numpy(test_attributes[join_data.columns[:num_features]])
        y_train = train_attributes[target]
        y_test = test_attributes[target]
    else:
        X_test = pd.DataFrame.to_numpy(join_data.columns[:num_features])
        y_test = join_data[target]
        ds_bops_histogram_test = ds_bops_histogram

    mlp = create_mlp(X_test.shape[1], regress=False)
    cnn1 = create_cnn(num_rows, num_columns, 1, regress=False)
    # cnn2 = models.create_cnn(num_rows, num_columns, 1, regress=False)
    # cnn3 = models.create_cnn(num_rows, num_columns, 1, regress=False)

    # combined_input = concatenate([mlp.output, cnn1.output, cnn2.output, cnn3.output])
    combined_input = concatenate([mlp.output, cnn1.output])

    x = Dense(4, activation="relu")(combined_input)
    x = Dense(1, activation="linear")(x)

    # model = Model(inputs=[mlp.input, cnn1.input, cnn2.input, cnn3.input], outputs=x)
    model = Model(inputs=[mlp.input, cnn1.input], outputs=x)

    EPOCHS = 40
    LR = 1e-2
    opt = Adam(lr=LR, decay=LR / EPOCHS)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

    if is_train:
        print ('Training the model')
        model.fit(
            [X_train, ds_bops_histogram_train], y_train,
            validation_data=([X_test, ds_bops_histogram_test], y_test),
            epochs=EPOCHS, batch_size=256)

        model.save(model_path)
        model.save_weights(model_weights_path)
    else:
        print ('Loading the saved model and model weights')
        model = load_model(model_path)
        model.load_weights(model_weights_path)

    print ('Testing')
    y_pred = model.predict([X_test, ds_bops_histogram_test])

    print ('r2 score: {}'.format(r2_score(y_test, y_pred)))

    diff = y_pred.flatten() - y_test
    percent_diff = (diff / y_test)
    abs_percent_diff = np.abs(percent_diff)

    # Compute the mean and standard deviation of the absolute percentage difference
    mean = np.mean(abs_percent_diff)
    std = np.std(abs_percent_diff)

    # NOTICE: mean is the MAPE value, which is the target we want to minimize
    print ('mean = {}, std = {}'.format(mean, std))
