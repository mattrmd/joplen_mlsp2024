# Written by ChatGPT4

import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model


class FeatureSelectionLayer(Layer):
    def __init__(self, n_features, l1_strength, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.l1_strength = l1_strength

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.n_features,),
            initializer="uniform",
            trainable=True,
        )

    def call(self, inputs):
        # Apply L1 regularization manually
        self.add_loss(self.l1_strength * tf.reduce_sum(tf.abs(self.kernel)))
        return inputs * self.kernel


class NN(BaseEstimator, ClassifierMixin):

    def __init__(
        self, hidden_layer_size, n_hidden_layers, activation, loss, sel_feat=False
    ):
        self.hidden_layer_size = hidden_layer_size
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.sel_feat = sel_feat
        self.loss = loss
        self.is_classifier = self.loss == "binary_crossentropy"
        self.output_activation = "sigmoid" if self.is_classifier else "linear"

    def build_model(self, input_shape, lambda_val=0, alpha=0):
        inputs = Input(shape=input_shape)
        x = inputs

        # Add feature selection layer if selected
        if self.sel_feat:
            self.feature_selection_layer = FeatureSelectionLayer(
                input_shape[0], lambda_val
            )
            x = self.feature_selection_layer(x)

        # Add hidden layers
        for _ in range(self.n_hidden_layers):
            x = Dense(
                self.hidden_layer_size,
                activation=self.activation,
                kernel_regularizer=tf.keras.regularizers.l2(alpha),
            )(x)

        # Output layer for regression
        outputs = Dense(1, activation=self.output_activation)(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=self.loss,
        )

    def fit(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs,
        patience,
        batch_size,
        learning_rate,
        lambda_val=0,
        alpha=0,
    ):
        self.learning_rate = learning_rate
        self.build_model(x_train.shape[1:], lambda_val=lambda_val, alpha=alpha)

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )

        device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"

        with tf.device(device):
            self.model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=0,
            )

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def score(self, X, y):
        y_pred = self.predict(X)
        if self.is_classifier:
            return tf.keras.losses.binary_crossentropy(y, y_pred).numpy()
        else:
            return tf.keras.losses.mean_squared_error(y, y_pred).numpy()

    def get_feature_selection_weights(self):
        if self.sel_feat and hasattr(self, "feature_selection_layer"):
            return self.feature_selection_layer.get_weights()[0]
        else:
            return None
