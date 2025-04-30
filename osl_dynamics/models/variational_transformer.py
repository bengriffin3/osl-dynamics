"""
Variational Multi-Scale Transformer for fMRI (TensorFlow Version)
- Input: voxelwise fMRI data Y (T x V)
- Output: reconstructed Y_hat, latent dynamics Z^l per scale l
- Interpretation:
    - Spatial attention = dynamic functional connectivity (dFC)
    - Temporal attention = HRF-like kernel or autocorrelation structure
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

class SpatialTemporalAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, N, D, heads):
        super().__init__()
        self.N = N
        self.D = D
        self.heads = heads
        self.dh = D // heads

        # Temporal attention projections
        self.q_proj_t = layers.Dense(D)
        self.k_proj_t = layers.Dense(D)
        self.v_proj_t = layers.Dense(D)

        # Spatial attention projections
        self.q_proj_s = layers.Dense(D)
        self.k_proj_s = layers.Dense(D)
        self.v_proj_s = layers.Dense(D)

        self.out_proj = layers.Dense(N)

    def call(self, x):
        # x: (T, N)
        T = tf.shape(x)[0]

        # Temporal attention (per region across time)
        x_t = tf.transpose(x, perm=[1, 0])  # (N, T)
        Q_t = self.q_proj_t(x_t)  # (N, D)
        K_t = self.k_proj_t(x_t)
        V_t = self.v_proj_t(x_t)

        Q_t = tf.reshape(Q_t, (self.N, self.heads, self.dh))
        K_t = tf.reshape(K_t, (self.N, self.heads, self.dh))
        V_t = tf.reshape(V_t, (self.N, self.heads, self.dh))

        attn_t = tf.nn.softmax(tf.matmul(Q_t, K_t, transpose_b=True) / tf.sqrt(float(self.dh)), axis=-1)
        Z_t = tf.matmul(attn_t, V_t)
        Z_t = tf.reshape(tf.transpose(Z_t, perm=[1, 0, 2]), (T, self.D))

        # Spatial attention (per time point across regions)
        Q_s = self.q_proj_s(x)
        K_s = self.k_proj_s(x)
        V_s = self.v_proj_s(x)

        Q_s = tf.reshape(Q_s, (T, self.heads, self.dh))
        K_s = tf.reshape(K_s, (T, self.heads, self.dh))
        V_s = tf.reshape(V_s, (T, self.heads, self.dh))

        attn_s = tf.nn.softmax(tf.matmul(Q_s, K_s, transpose_b=True) / tf.sqrt(float(self.dh)), axis=-1)
        Z_s = tf.matmul(attn_s, V_s)
        Z_s = tf.reshape(tf.transpose(Z_s, perm=[1, 0, 2]), (T, self.D))

        Z = Z_t + Z_s
        return self.out_proj(Z), attn_t, attn_s

class VariationalEncoder(tf.keras.Model):
    def __init__(self, V, P_list, D=64, heads=4):
        super().__init__()
        self.P_list = P_list  # list of tf.Tensor (V x N_l)
        self.attn_blocks = []
        self.mu_heads = []
        self.logvar_heads = []

        for P in P_list:
            N_l = P.shape[1]
            self.attn_blocks.append(SpatialTemporalAttentionBlock(N_l, D, heads))
            self.mu_heads.append(layers.Dense(N_l))
            self.logvar_heads.append(layers.Dense(N_l))

    def call(self, Y):
        # Y: (T, V)
        recons = []
        kldivs = []
        spatial_attns = []
        temporal_attns = []

        for i, P in enumerate(self.P_list):
            A = tf.matmul(Y, P)  # (T, N_l)
            Z, attn_t, attn_s = self.attn_blocks[i](A)

            mu = self.mu_heads[i](Z)
            logvar = self.logvar_heads[i](Z)
            std = tf.exp(0.5 * logvar)
            eps = tf.random.normal(tf.shape(std))
            Z_sample = mu + eps * std

            Y_hat = tf.matmul(Z_sample, tf.transpose(P))  # (T, V)
            recons.append(Y_hat)

            kldiv = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))
            kldivs.append(kldiv)

            spatial_attns.append(attn_s)
            temporal_attns.append(attn_t)

        Y_hat_all = tf.add_n(recons) / len(recons)
        total_kldiv = tf.add_n(kldivs)
        return Y_hat_all, total_kldiv, spatial_attns, temporal_attns