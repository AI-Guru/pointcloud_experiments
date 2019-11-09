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


class GraphAttention(tf.keras.layers.Layer):
    """
    the single-head GAPLayer learns self-attention and neighboring-attention
    features in parallel that are then fused together by a non-linear activation
    function leaky RELU to obtain attention coefficients, which are further
    normalized by a softmax function, then a linear combination operation is
    applied to finally generate attention feature. MLP{} denotes multi-layer
    perceptron operation, numbers in brace stand for size of a set of filters,
    and we use the same notation for the remainder.
    """

    def __init__(self, k, features, bn_decay=None, **kwargs):

        self.k = k
        self.features = features
        #self.bn_decay + bn_decay
        super(GraphAttention, self).__init__(**kwargs)


    def build(self, input_shape):

        super(GraphAttention, self).build(input_shape)


    def call(self, input):

        point_cloud = input[0]
        knn = input[1]

        # Pointcloud stream.
        pc_mlp1 = layers.Dense(self.features, activation="relu", name=self.name + "_pc_mlp1")(point_cloud)
        pc_mlp2 = layers.Dense(1, activation="relu", name=self.name + "_pc_mlp2")(pc_mlp1)

        # KNN stream.
        knn_mlp1 = layers.Dense(self.features, activation="relu", name=self.name + "_graph_features")(knn)
        graph_features = knn_mlp1
        knn_mlp2 = layers.Dense(1, activation="relu", name=self.name + "_knn_mlp2")(knn_mlp1)

        # Compute attention coefficients.
        # Add point_cloud_mlp2 and point_cloud_mlp2 and apply softmax.
        attention_coefficients = layers.Lambda(lambda x: tf.add(
            tf.expand_dims(x[0], axis=-1),
            x[1]
        ), name=self.name + "_add")([pc_mlp2, knn_mlp2])
        attention_coefficients = layers.Permute([1, 3, 2])(attention_coefficients)
        attention_coefficients = layers.LeakyReLU()(attention_coefficients)
        attention_coefficients = layers.Activation("softmax", name=self.name + "_attention_coefficients")(attention_coefficients)
        #print(attention_coefficients)
        #attention_coefficients_shape = (K.int_shape(point_cloud)[1], self.k,)
        #print(attention_coefficients_shape)
        #attention_coefficients = layers.Reshape(attention_coefficients_shape)(attention_coefficients)
        #print(attention_coefficients)
        # Compute attention features.
        # Matmul graph-features and attention coeeficients. Apply squeeze.
        attention_features = layers.Lambda(
            lambda x: tf.matmul(x[1], x[0])
        )([graph_features, attention_coefficients])
        attention_features = layers.Lambda(
            lambda x: K.squeeze(x, axis=2),
            name=self.name + "_attention_features"
        )(attention_features)

        return attention_features, graph_features, attention_coefficients


class MultiGraphAttention(tf.keras.layers.Layer):
    """
    The GAPLayer with M heads, as shown in 2(a) , takes N points with F dimensions as input and concatenates attention feature and graph feature respectively from all heads to generate multi-attention features and multi-graph features as output.
    """

    def __init__(self, k, features, heads, bn_decay=None, **kwargs):

        self.k = k
        self.features = features
        self.heads = heads
        #self.bn_decay + bn_decay
        super(MultiGraphAttention, self).__init__(**kwargs)


    def call(self, input):

        # Input for a pointcloud.
        point_cloud = input

        # Create the KNN layer and apply it to the input.
        knn = KNN(k=self.k, name=self.name + "_knn")(point_cloud)

        # Do multi-head attention.
        attention_features_list = []
        graph_features_list = []
        attention_coefficients_list = []
        for head_index in range(self.heads):
            graph_attention = GraphAttention(k=self.k, features=self.features)([point_cloud, knn])
            attention_features = graph_attention[0]
            graph_features = graph_attention[1]
            attention_coefficients = graph_attention[2]

            attention_features_list.append(attention_features)
            graph_features_list.append(graph_features)
            attention_coefficients_list.append(attention_coefficients)

        # Create all attention features. Includes skip connection to input.
        if self.heads == 1:
            multi_attention_features_shape = (
                K.int_shape(attention_features_list[0])[1],
                1,
                K.int_shape(attention_features_list[0])[2]
            )
            multi_attention_features = layers.Reshape(multi_attention_features_shape, name=self.name + "_multi_attention_features")(attention_features_list[0])
        else:
            multi_attention_features = layers.Lambda(lambda x: K.stack(x, axis=2), name=self.name + "_multi_attention_features")(attention_features_list)

        # Create all graph features.
        if self.heads == 1:
            multi_graph_features_shape = (
                K.int_shape(graph_features_list[0])[1],
                1,
                K.int_shape(graph_features_list[0])[2],
                K.int_shape(graph_features_list[0])[3]
            )
            multi_graph_features = layers.Reshape(multi_graph_features_shape, name=self.name + "_multi_graph_features")(graph_features_list[0])
        else:
            multi_graph_features = layers.Lambda(lambda x: K.stack(x, axis=2), name=self.name + "_multi_graph_features")(graph_features_list)

        # Create all graph features.
        if self.heads == 1:
            multi_attention_coefficients = attention_coefficients_list[0]
        else:
            multi_attention_coefficients = layers.concatenate(attention_coefficients_list, name=self.name + "_multi_graph_features", axis=2)

        return multi_attention_features, multi_graph_features, multi_attention_coefficients


class Transform(tf.keras.layers.Layer):
    """
    spatial transform network: The spatial transform network is used to make
    point cloud invariant to certain transformations. The model learns a 3 × 3
    matrix for affine transformation from a single-head GAPLayer with 16
    channels.
    """

    def __init__(self, k, features, bn_decay=None, **kwargs):

        self.k = k
        self.features = features

        # Weights for the learned transformation matrix.
        w_init = tf.zeros_initializer()
        self.transform_w = tf.Variable(
            initial_value=w_init(shape=(256, 3 * 3), dtype='float32'),
            trainable=True
        )

        # Biases for the learned transformation matrix.
        b_init = tf.zeros_initializer()
        self.transform_b = tf.Variable(
            initial_value=b_init(shape=(3 * 3,), dtype='float32'),
            trainable=True
        )

        super(Transform, self).__init__(**kwargs)


    def build(self, input_shape):
        #assert len(input_shape) == 2
        #pointcloud, knn = input_shape

        #self.dense_pc_1 = layers.Dense()


        super(Transform, self).build(input_shape)


    def call(self, input):

        point_cloud = input
        number_of_points = K.int_shape(point_cloud)[1]
        print(point_cloud)

        # Use a single-head Graph-Attention layer.
        attention_features, graph_features, attention_coefficients = MultiGraphAttention(k=self.k, features=self.features, heads=1)(point_cloud)

        # TODO implementation has a skip connection... check this...

        # edge_feature in original is our attention_feature
        #edge_feature = ???
        # TODO max of graph features
        #locals_max_transform = ???
        attention_max_transform = layers.Lambda(lambda x: tf.reduce_max(x, axis=-2, keepdims=True))(graph_features)

        # TODO BN decay?
        # Dense on attention features.
        net = layers.Dense(64, activation="linear")(attention_features)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)

        # Dense on attention features.
        net = layers.Dense(128, activation="linear")(attention_features)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)

        # Concatenate with max.
        net = layers.concatenate([net, attention_max_transform])

        # Dense on attention features.
        net = layers.Dense(1024, activation="linear")(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)

        # Apply max pooling.
        # TODO: Consider using global pooling layer.
        net = layers.MaxPooling2D((number_of_points, 1))(net)

        # Flatten.
        net = layers.Flatten()(net)

        # Fully connected.
        net = layers.Dense(512, activation="linear")(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)

        # Fully connected.
        net = layers.Dense(512, activation="linear")(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)

        # Compute the transformation matrix.
        # TODO Why do we do that?
        self.transform_b += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, self.transform_w)
        transform = tf.nn.bias_add(transform, self.transform_b)

        # Turn transform into a 3x3 matrix.
        transform = layers.Reshape((3, 3))(transform)

        print("transform", transform.shape, transform)
        print("point_cloud", point_cloud.shape, point_cloud)

        # Final transformation of the pointcloud.
        result = layers.Lambda(lambda x: K.dot(x[0], x[1]))(point_cloud, transform)
        return result


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
