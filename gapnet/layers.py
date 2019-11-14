import tensorflow as tf
#assert tf.__version__.startswith("1.1"), "Expected tensorflow 1.1X, got {}".format(tf.__version__)
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

    def __init__(self, features_out, batch_normalization=True, **kwargs):

        self.features_out = features_out
        self.batch_normalization = batch_normalization=True

        # Call super.
        super(GraphAttention, self).__init__(**kwargs)


    def build(self, input_shapes):

        assert len(input_shapes) == 2

        point_cloud_shape = input_shapes[0].as_list()
        self.number_of_points = point_cloud_shape[1]
        self.features_in = point_cloud_shape[-1]

        knn_shape = input_shapes[1].as_list()
        assert knn_shape[1] == point_cloud_shape[1]
        self.k = knn_shape[2]

        # MLP 1 for self attention.
        self.self_attention_mlp1 = layers.Dense(
            self.features_out,
            activation="relu",
            name=self.name + "_self_attention_mlp1"
            )
        if self.batch_normalization == True:
            self.self_attention_bn1 = layers.BatchNormalization()

        # MLP 2 for self attention.
        self.self_attention_mlp2 = layers.Dense(
            1,
            activation="relu",
            name=self.name + "_self_attention_mlp2"
            )
        if self.batch_normalization == True:
            self.self_attention_bn2 = layers.BatchNormalization()

        # MLP 1 for neighbor attention.
        self.neighbor_attention_mlp1 = layers.Dense(
            self.features_out,
            activation="relu",
            name=self.name + "_neighbor_attention_mlp1"
            )
        if self.batch_normalization == True:
            self.neighbor_attention_bn1 = layers.BatchNormalization()

        # MLP 2 for neighbor attention.
        self.neighbor_attention_mlp2 = layers.Dense(
            1,
            activation="relu",
            name=self.name + "_neighbor_attention_mlp2"
            )
        if self.batch_normalization == True:
            self.neighbor_attention_bn2 = layers.BatchNormalization()

        # Final bias.
        self.output_bias = self.add_variable(
            "kernel",
            shape=[self.number_of_points, 1, self.features_out])

        # Call super.
        super(GraphAttention, self).build(input_shapes)


    def call(self, inputs):

        # The first part of the input is the pointcloud.
        point_cloud = inputs[0]
        assert_shape_is(point_cloud, (1024, 3))

        # The second part of the input are the KNNs.
        knn = inputs[1]
        assert_shape_is(knn, (1024, 20, 3))

        # Reshape the pointcloud if necessary.
        if len(point_cloud.shape) == 4:
            pass
        elif len(point_cloud.shape) == 3:
            point_cloud = K.expand_dims(point_cloud, axis=2)
        else:
            raise Exception("Invalid shape!")
        assert_shape_is(point_cloud, (1024, 1, 3))

        # Tile the pointcloud to make it compatible with KNN.
        point_cloud_tiled = K.tile(point_cloud, [1, 1, self.k, 1])
        assert_shape_is(point_cloud_tiled, (1024, 20, 3))

        # Compute difference between tiled pointcloud and knn.
        point_cloud_knn_difference = point_cloud_tiled - knn
        assert_shape_is(point_cloud_knn_difference, (1024, 20, 3))

        # MLP 1 for self attention including batch normalization.
        self_attention = self.self_attention_mlp1(point_cloud)
        if self.batch_normalization == True:
            self_attention = self.self_attention_bn1(self_attention)
        assert_shape_is(self_attention, (1024, 1, 16))

        # MLP 2 for self attention including batch normalization.
        self_attention = self.self_attention_mlp2(self_attention)
        if self.batch_normalization == True:
            self_attention = self.self_attention_bn2(self_attention)
        assert_shape_is(self_attention, (1024, 1, 1))

        # MLP 1 for neighbor attention including batch normalization.
        neighbor_attention = self.neighbor_attention_mlp1(point_cloud_knn_difference)
        if self.batch_normalization == True:
            neighbor_attention = self.neighbor_attention_bn1(neighbor_attention)
        assert_shape_is(neighbor_attention, (1024, 20, 16))

        # Graph features are the ouput of the first MLP.
        graph_features = neighbor_attention

        # MLP 2 for neighbor attention including batch normalization.
        neighbor_attention = self.neighbor_attention_mlp2(neighbor_attention)
        if self.batch_normalization == True:
            neighbor_attention = self.neighbor_attention_bn2(neighbor_attention)
        assert_shape_is(neighbor_attention, (1024, 20, 1))

        # Merge self attention and neighbor attention to get attention coefficients.
        logits = self_attention + neighbor_attention
        assert_shape_is(logits, (1024, 20, 1))
        logits = K.permute_dimensions(logits, (0, 1, 3, 2))
        assert_shape_is(logits, (1024, 1, 20))

        # Apply leaky relu and softmax to logits to get attention coefficents.
        logits = K.relu(logits, alpha=0.2)
        attention_coefficients = K.softmax(logits)
        assert_shape_is(attention_coefficients, (1024, 1, 20))

        # Compute attention features from attention coefficients and graph features.
        attention_features = tf.matmul(attention_coefficients, graph_features)
        attention_features = tf.add(attention_features, self.output_bias)
        attention_features = K.relu(attention_features)
        assert_shape_is(attention_features, (1024, 1, 16))

        # Reshape graph features.
        #graph_features = K.expand_dims(graph_features, axis=2)
        assert_shape_is(graph_features, (1024, 20, 16))

        # Done.
        return attention_features, graph_features, attention_coefficients


class MultiGraphAttention(tf.keras.layers.Layer):
    """
    The GAPLayer with M heads, as shown in 2(a) , takes N points with F dimensions as input and concatenates attention feature and graph feature respectively from all heads to generate multi-attention features and multi-graph features as output.
    """

    def __init__(self, k, features_out, heads, batch_normalization=True, **kwargs):

        self.k = k
        self.features_out = features_out
        self.heads = heads
        self.batch_normalization = batch_normalization
        #self.bn_decay + bn_decay

        # Call super.
        super(MultiGraphAttention, self).__init__(**kwargs)


    def build(self, input_shape):

        self.graph_attentions = [GraphAttention(features_out=self.features_out, batch_normalization=self.batch_normalization) for _ in range(self.heads)]

        # Call super.
        super(MultiGraphAttention, self).build(input_shape)


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
            graph_attention = self.graph_attentions[head_index]([point_cloud, knn])
            attention_features = graph_attention[0]
            graph_features = graph_attention[1]
            attention_coefficients = graph_attention[2]

            attention_features_list.append(attention_features)
            graph_features_list.append(graph_features)
            attention_coefficients_list.append(attention_coefficients)

        # Only one head. Return first element of lists.
        if self.heads == 1:
            multi_attention_features = attention_features_list[0]
            multi_graph_features = graph_features_list[0]
            multi_attention_coefficients = attention_coefficients_list[0]

        # More than one head. Stack.
        else:
            multi_attention_features = K.concatenate(attention_features_list, axis=3)
            multi_graph_features = K.concatenate(graph_features_list, axis=3)
            multi_attention_coefficients = K.concatenate(attention_coefficients_list, axis=3)

        assert_shape_is(multi_attention_features, (1024, 1, 16 * self.heads))
        assert_shape_is(multi_graph_features, (1024, 20, 16 * self.heads))
        assert_shape_is(multi_attention_coefficients, (1024, 1, 20 * self.heads))

        # Done.
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


def assert_shape_is(tensor, expected_shape):
    assert isinstance(tensor, tf.Tensor), type(tensor)
    assert isinstance(expected_shape, list) or isinstance(expected_shape, tuple), type(expected_shape)
    tensor_shape = tensor.shape[1:]
    if tensor_shape != expected_shape:
        raise Exception("{} is not equal {}".format(tensor_shape, expected_shape))
