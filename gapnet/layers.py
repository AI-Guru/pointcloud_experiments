import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import models, layers

class KNN(tf.keras.layers.Layer):
    """
    For a given sequence of vectors, computes the k-nearest neighbors.
    """

    def __init__(self, k, **kwargs):
        self.k = k
        super(KNN, self).__init__(**kwargs)


    def build(self, input_shape):

        super(KNN, self).build(input_shape)


    def call(self, input):

        point_cloud = input

        point_cloud_transpose = K.permute_dimensions(point_cloud, [0, 2, 1])

        # Compute distances.
        point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
        point_cloud_inner = -2 * point_cloud_inner
        point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keepdims=True)
        point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
        adj_matrix = point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

        # Compute indices.
        neg_adj = -adj_matrix
        _, nn_idx = tf.nn.top_k(neg_adj, k=self.k)

        # Compute the neighbors.
        batch_size = tf.shape(point_cloud)[0] # Note: Treat batch-size differently.
        num_points = point_cloud.get_shape()[1]
        num_dims = point_cloud.get_shape()[2]
        idx_ = tf.range(batch_size) * num_points
        idx_ = tf.reshape(idx_, [-1, 1, 1])
        point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
        point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)

        return point_cloud_neighbors



class Transform(tf.keras.layers.Layer):

    def __init__(self, k, bn_decay=None, **kwargs):

        self.k = k
        #self.bn_decay + bn_decay
        super(Transform, self).__init__(**kwargs)


    def build(self, input_shape):
        #assert len(input_shape) == 2
        #pointcloud, knn = input_shape

        #self.dense_pc_1 = layers.Dense()


        super(Transform, self).build(input_shape)


    def call(self, input):

        point_cloud = input[0]
        attention_features = input[1]
        graph_features = input[2]

        attention_max_transform = tf.reduce_max(attention_features, axis=-2, keepdims=True)

        # Start with. graph_features
        net = layers.Dense(64)(graph_features)
        net = layers.Dense(128)(net)

        concatenate = tf.concat([net, attention_max_transform], axis=-1)
        net = layers.Dense(128)(concatenate)
        net = layers.MaxPooling2D()(net)

        net = layers.Flatten()(net)
        net = layers.Dense(512)(concatenate)
        net = layers.Dense(256)(concatenate)

        return net


        # TODO multiply
        #point_cloud_transformed = layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([point_cloud, transform])
        #(32, 1024, 3)


        print(type(input))
        # Input must be point-cloud and knn.
        assert isinstance(input, list)
        assert len(input) == 2
        point_cloud, knn = input

        print(point_cloud.shape, knn.shape)
        return knn, knn, knn
