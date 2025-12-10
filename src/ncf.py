# src/ncf.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import List

class NeuralCollaborativeFiltering:
    """
    Neural Collaborative Filtering (NCF) model combining GMF and MLP paths.
    This version supports configurable learning rates and layer-specific dropout rates.
    """
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 mlp_layers: List[int] = [128, 64, 32],
                 dropout_rates: List[float] = [0.2, 0.2, 0.2], # Default to a list
                 l2_reg: float = 1e-6, use_gmf: bool = True, use_mlp: bool = True,
                 learning_rate: float = 0.001):
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.mlp_layers = mlp_layers
        self.dropout_rates = dropout_rates
        self.l2_reg = float(l2_reg)
        self.use_gmf = use_gmf
        self.use_mlp = use_mlp
        self.learning_rate = learning_rate
        self.model = None

        if len(self.mlp_layers) != len(self.dropout_rates):
            raise ValueError("The number of MLP layers must match the number of dropout rates.")

    def build(self) -> Model:
        """Builds the Keras model architecture."""
        user_input = layers.Input(shape=(1,), name='user_input')
        item_input = layers.Input(shape=(1,), name='item_input')

        # --- GMF Path ---
        if self.use_gmf:
            gmf_user_embedding = layers.Embedding(self.n_users, self.embedding_dim, embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg), name='gmf_user_embedding')(user_input)
            gmf_item_embedding = layers.Embedding(self.n_items, self.embedding_dim, embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg), name='gmf_item_embedding')(item_input)
            gmf_user_vec = layers.Flatten()(gmf_user_embedding)
            gmf_item_vec = layers.Flatten()(gmf_item_embedding)
            gmf_vec = layers.Multiply()([gmf_user_vec, gmf_item_vec])
        else:
            gmf_vec = None

        # --- MLP Path ---
        if self.use_mlp:
            mlp_user_embedding = layers.Embedding(self.n_users, self.embedding_dim, embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg), name='mlp_user_embedding')(user_input)
            mlp_item_embedding = layers.Embedding(self.n_items, self.embedding_dim, embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg), name='mlp_item_embedding')(item_input)
            mlp_user_vec = layers.Flatten()(mlp_user_embedding)
            mlp_item_vec = layers.Flatten()(mlp_item_embedding)
            mlp_concat = layers.Concatenate()([mlp_user_vec, mlp_item_vec])
            
            x = mlp_concat
            # Use the list of dropout rates for each corresponding MLP layer
            for i, units in enumerate(self.mlp_layers):
                x = layers.Dense(units, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(self.dropout_rates[i])(x)
            mlp_vec = x
        else:
            mlp_vec = None

        # --- Combine Paths ---
        if self.use_gmf and self.use_mlp:
            concat_vec = layers.Concatenate()([gmf_vec, mlp_vec])
        elif self.use_gmf:
            concat_vec = gmf_vec
        else:
            concat_vec = mlp_vec

        output = layers.Dense(1, activation='linear', name='output')(concat_vec)
        
        model = Model(inputs=[user_input, item_input], outputs=output)
        
        # Use the configurable learning rate
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )
        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=1024, verbose=1):
        """Trains the model with early stopping and learning rate reduction."""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        history = self.model.fit(
            [X_train[:, 0], X_train[:, 1]], y_train,
            validation_data=([X_val[:, 0], X_val[:, 1]], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        return history

    def save(self, path):
        """Saves the Keras model."""
        self.model.save(path)