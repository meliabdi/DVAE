import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.layers import Lambda, Layer, Input, Dense, GaussianNoise
from sklearn.metrics.pairwise import cosine_similarity




class Sampling(Layer):
     def call(self, inputs):
         z_mean, z_log_var = inputs
         batch = K.shape(z_mean)[0]
         dim = K.int_shape(z_mean)[1]
         epsilon = K.random_normal(shape=(batch, dim))
         return z_mean + K.exp(0.5 * z_log_var) * epsilon

 class VAELoss(Layer):
     def call(self, inputs):
         x, x_decoded_mean, z_mean, z_log_var = inputs
         reconstruction_loss = tf.reduce_mean(tf.keras.losses.MSE(x, x_decoded_mean))
         kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
         kl_loss = K.sum(kl_loss, axis=-1)
         kl_loss *= -0.5
         total_loss = K.mean(reconstruction_loss + kl_loss)
         self.add_loss(total_loss)
         return x_decoded_mean
    
     def compute_output_shape(self, input_shape):
         return input_shape[1]

 def build_deep_variational_autoencoder(input_dim, latent_dim):
     inputs = Input(shape=(input_dim,))
     h = Dense(256, activation='tanh')(inputs)
     h = Dense(128, activation='tanh')(h)
     h = Dense(64, activation='tanh')(h)
     z_mean = Dense(latent_dim)(h)
     z_log_var = Dense(latent_dim)(h)
     z = Sampling()([z_mean, z_log_var])

     decoder_h = Dense(64, activation='tanh')
     decoder_h2 = Dense(128, activation='tanh')
     decoder_h3 = Dense(256, activation='tanh')
     decoder_mean = Dense(input_dim, activation='tanh')
     h_decoded = decoder_h(z)
     h_decoded = decoder_h2(h_decoded)
     h_decoded = decoder_h3(h_decoded)
     x_decoded_mean = decoder_mean(h_decoded)

     vae_loss = VAELoss()([inputs, x_decoded_mean, z_mean, z_log_var])
     vae = models.Model(inputs, vae_loss)
     vae.compile(optimizer='adam')
     encoder = models.Model(inputs, z_mean)
     return vae, encoder








def calculate_cosine_similarity(autoencoder, patient_data, drug_data):
     similarities = []
     for i, drug_name in enumerate(filtered_drug_df.columns[1:]):
         drug_profile = drug_data[i, :].reshape(1, -1)
         encoded_drug_data = autoencoder.predict(drug_profile)
         similarity = cosine_similarity(patient_data, encoded_drug_data)
         similarities.append((drug_name, similarity[0][0]))
     similarities.sort(key=lambda x: x[1])  # ascending order
     return similarities
