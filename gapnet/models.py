import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras import backend as K
from .layers import KNN, Transform, MultiGraphAttention, GraphAttention
from .utils import assert_shape_is


class GAPNet(tf.keras.Model):

    def __init__(self, number_of_points=1024, features_in=3, k=20, features_out=16, **kwargs):
        super(GAPNet, self).__init__()

        self.number_of_points = number_of_points
        self.features_in = features_in
        self.k = k
        self.features_out = features_out

        self.build_graph(input_shape=(None, number_of_points, features_in))


    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)

    def build(self, input_shape, **kwargs):

                # Input for a pointcloud.

        # Create attention layer with one head.
        self.onehead_attention = MultiGraphAttention(k=self.k, features_out=self.features_out, heads=1)

        # Create spatial transform layer.
        self.transform = Transform()

        # Create attention layer with four heads.
        self.fourhead_attention = MultiGraphAttention(k=self.k, features_out=self.features_out, heads=4)

        # MLP 1 on attention features.
        self.mlp1 = layers.Dense(64, activation="linear")
        self.mlp_bn1 = layers.BatchNormalization()
        self.mlp_activation1 = layers.Activation("relu")

        # MLP 2 on attention features.
        self.mlp2 = layers.Dense(64, activation="linear")
        self.mlp_bn2 = layers.BatchNormalization()
        self.mlp_activation2 = layers.Activation("relu")

        # MLP 3 on attention features.
        self.mlp3 = layers.Dense(64, activation="linear")
        self.mlp_bn3 = layers.BatchNormalization()
        self.mlp_activation3 = layers.Activation("relu")

        # MLP 4 on attention features.
        self.mlp4 = layers.Dense(128, activation="linear")
        self.mlp_bn4 = layers.BatchNormalization()
        self.mlp_activation4 = layers.Activation("relu")

        # MLP 5.
        self.mlp5 = layers.Dense(1024, activation="linear")
        self.mlp_bn5 = layers.BatchNormalization()
        self.mlp_activation5 = layers.Activation("relu")

        # Flatten.
        self.flatten = layers.Flatten()

        # Dense 1.
        self.dense1 = layers.Dense(512, activation="linear")
        self.dense_dropout1 = layers.Dropout(0.5)

        # Dense 1.
        self.dense2 = layers.Dense(256, activation="linear")
        self.dense_dropout2 = layers.Dropout(0.5)

        # Dense 1.
        self.dense3 = layers.Dense(40, activation="softmax")

        super(GAPNet, self).build(input_shape)


    def call(self, inputs):

        point_cloud = inputs
        assert_shape_is(point_cloud, (1024, 3))

        # First attention layer with one head.
        onehead_attention = self.onehead_attention(point_cloud)
        onehead_attention_features = onehead_attention[0]
        onehead_graph_features = onehead_attention[1]
        onehead_attention_coefficients = onehead_attention[2]
        assert_shape_is(onehead_attention_features, (1024, 1, 16))
        assert_shape_is(onehead_graph_features, (1024, 20, 16))
        assert_shape_is(onehead_attention_coefficients, (1024, 1, 20))

        # Skip connection from point cloud to attention features.
        point_cloud_expanded = K.expand_dims(point_cloud, axis=2)
        assert_shape_is(point_cloud_expanded, (1024, 1, 3))
        onehead_attention_features = K.concatenate([onehead_attention_features, point_cloud_expanded])
        assert_shape_is(onehead_attention_features, (1024, 1, 19))
        del point_cloud_expanded

        # Spatial transform.
        point_cloud_transformed = self.transform([point_cloud, onehead_attention_features, onehead_graph_features])
        assert_shape_is(point_cloud_transformed, (1024, 3))
        del point_cloud

        # Second attention layer with four head.
        fourhead_attention = self.fourhead_attention(point_cloud_transformed)
        fourhead_attention_features = fourhead_attention[0]
        fourhead_graph_features = fourhead_attention[1]
        fourhead_attention_coefficients = fourhead_attention[2]
        assert_shape_is(fourhead_attention_features, (1024, 1, 64))
        assert_shape_is(fourhead_graph_features, (1024, 20, 64))
        assert_shape_is(fourhead_attention_coefficients, (1024, 1, 80))

        # Skip connection from transformed point cloud to attention features.
        point_cloud_expanded = K.expand_dims(point_cloud_transformed, axis=2)
        assert_shape_is(point_cloud_expanded, (1024, 1, 3))
        onehead_attention_features = K.concatenate([fourhead_attention_features, point_cloud_expanded])
        assert_shape_is(onehead_attention_features, (1024, 1, 67))

        # MLP 1 on attention features.
        net1 = self.mlp1(onehead_attention_features)
        net1 = self.mlp_bn1(net1)
        net1 = self.mlp_activation1(net1)
        assert_shape_is(net1, (1024, 1, 64))

        # MLP 2 on attention features.
        net2 = self.mlp2(net1)
        net2 = self.mlp_bn2(net2)
        net2 = self.mlp_activation2(net2)
        assert_shape_is(net2, (1024, 1, 64))

        # MLP 3 on attention features.
        net3 = self.mlp3(net2)
        net3 = self.mlp_bn3(net3)
        net3 = self.mlp_activation3(net3)
        assert_shape_is(net3, (1024, 1, 64))

        # MLP 4 on attention features.
        net4 = self.mlp4(net3)
        net4 = self.mlp_bn4(net4)
        net4 = self.mlp_activation4(net4)
        assert_shape_is(net4, (1024, 1, 128))

        # Maximum for graph features.
        fourhead_graph_features_max = tf.reduce_max(fourhead_graph_features, axis=2, keepdims=True)
        assert_shape_is(fourhead_graph_features_max, (1024, 1, 64))

        # Concatenate all MLPs and maximum of graph features.
        net = layers.concatenate([net1, net2, net3, net4, fourhead_graph_features_max])
        assert_shape_is(net, (1024, 1, 384))

        # MLP 5.
        net = self.mlp5(net)
        net = self.mlp_bn5(net)
        net = self.mlp_activation5(net)
        assert_shape_is(net, (1024, 1, 1024))

        # Maximum for net.
        net = K.max(net, axis=1, keepdims=True)
        assert_shape_is(net, (1, 1, 1024))

        # Flatten.
        net = self.flatten(net)
        assert_shape_is(net, (1024,))

        # Dense 1.
        net = self.dense1(net)
        net = self.dense_dropout1(net)
        assert_shape_is(net, (512,))

        # Dense 2.
        net = self.dense2(net)
        net = self.dense_dropout2(net)
        assert_shape_is(net, (256,))

        # Dense 3.
        net = self.dense3(net)
        assert_shape_is(net, (40,))

        return net





def create_gapnet_dev(number_of_points, nb_classes):
    features = 3
    k = 20
    heads = 4
    features_out = 16

    # Input for a pointcloud.
    point_cloud = layers.Input(shape=(number_of_points, features))

    # Create the Graph Attention.
    multi_graph_attention = MultiGraphAttention(k=k, features_out=features_out, heads=heads, batch_normalization=True)(point_cloud)

    # For now: Flatten
    attention_features = multi_graph_attention[0]
    attention_features = layers.Flatten()(attention_features)

    # For now: Flatten.
    graph_features = multi_graph_attention[1]
    graph_features = layers.Flatten()(graph_features)

    # For now: Merge all.
    merged = layers.Concatenate()([attention_features, graph_features])

    output = merged

    # Final layer. Classifier.
    output = layers.Dense(nb_classes, activation="softmax")(output)

    # Create the model.
    model = models.Model(point_cloud, output)
    return model

def build_model_paper(
    number_of_points,
    features,
    k,
    heads1,
    heads2,
    bn_decay,
    output_size,
    output_activation):

    # Create the input shape.
    point_cloud = layers.Input(shape=(number_of_points, 3), name="point_cloud_input")
    print("point_cloud", point_cloud)

    # Then apply spatial transformation.
    point_cloud_transformed = TransformPaper(k=k)(point_cloud)
    print("point_cloud_transformed", point_cloud_transformed)

    return models.Model(point_cloud, point_cloud_transformed), None



    # Create attention on the input.
    print("Attention before transformation.")
    multi_attention_features, all_graph_features = build_multi_head_attention(point_cloud, k, heads=4, features=features, name="attention1")
    print("")


    # TODO What is this? skip connection?! neighbors_features in original
    #point_cloud_expanded = layers.Reshape((K.int_shape(point_cloud)[1], 1, K.int_shape(point_cloud)[2]))(point_cloud)
    #print("point_cloud_expanded", point_cloud_expanded.shape)
    #print("multi_attention_features before", multi_attention_features.shape)
    #multi_attention_features = layers.concatenate([multi_attention_features, point_cloud_expanded], name=name + "_multi_attention_features_with_skip")
    #print("multi_attention_features after", multi_attention_features.shape)

    # Local transformation. TODO
    point_cloud_transformed = point_cloud
    #transform = Transform(k=3)([point_cloud, multi_attention_features, all_graph_features])

    # Create KNN for transformed pointcloud.
    #print("Attention after transformation.")
    #multi_attention_features_transformed, all_graph_features_transformed = build_multi_head_attention(point_cloud_transformed, k, heads2, name="attention2")
    #print("")
    #print(multi_attention_features_transformed.shape, all_graph_features_transformed.shape)

    # TODO Convolutions.


    #assert bn_decay is None
    #for hidden in [512, 256]:
    #    y = layers.Dense(hidden)(y)
    #    y = layers.BatchNormalization()(y) # TODO bn decay
    #    y = layers.Activation("relu")(y)
    #    y = layers.Dropout(0.5)(y)

    #y = layers.Dense(output_size, output_activation)(y)

    # TODO Fix this.
    #output = knn
    #output = [attention_features, graph_features, attention_coefficients]
    output = [multi_attention_features, all_graph_features]
    #output = point_cloud_transformed

    #output = transform

    # TODO also create attention
    prediction_model = models.Model(point_cloud, output)
    attention_model = models.Model(point_cloud, output)

    return prediction_model, attention_model


def build_multi_head_attention(point_cloud, k, heads, features, name):

    # Create KNN for input.
    knn = KNN(k=k, name=name + "_knn")(point_cloud)

    attention_features_list = []
    graph_features_list = []
    for head_index in range(heads):
        attention_features, graph_features, attention_coefficients = build_attention(
            point_cloud,
            knn,
            features=features,
            name=name + "_head_{}".format(head_index)
        )

        attention_features_list.append(attention_features)
        graph_features_list.append(graph_features)

    # Create all attention features. Includes skip connection to input.
    if heads == 1:
        multi_attention_features_shape = (
            K.int_shape(attention_features_list[0])[1],
            1,
            K.int_shape(attention_features_list[0])[2]
        )
        multi_attention_features = layers.Reshape(multi_attention_features_shape, name=name + "_multi_attention_features")(attention_features_list[0])
    else:
        multi_attention_features = layers.concatenate(attention_features_list, name=name + "_multi_attention_features")

    # Create all graph features.
    if heads == 1:
        multi_graph_features_shape = (
            K.int_shape(graph_features_list[0])[1],
            K.int_shape(graph_features_list[0])[2],
            1,
            K.int_shape(graph_features_list[0])[3]
        )
        multi_graph_features = layers.Reshape(multi_graph_features_shape, name=name + "_multi_graph_features")(graph_features_list[0])
    else:
        multi_graph_features = layers.concatenate(graph_features_list, name=name + "_multi_graph_features")
    print("multi_graph_features", multi_graph_features.shape)

    return multi_attention_features, multi_graph_features


def build_attention(point_cloud, knn, features, name):

    # TODO Both streams are CNNs in the original implementation. What to do?

    # Pointcloud stream.
    pc_mlp1 = layers.Dense(features, activation="relu", name=name + "_pc_mlp1")(point_cloud)
    pc_mlp2 = layers.Dense(1, activation="relu", name=name + "_pc_mlp2")(pc_mlp1)

    # KNN stream.
    knn_mlp1 = layers.Dense(features, activation="relu", name=name + "_graph_features")(knn)
    graph_features = knn_mlp1
    knn_mlp2 = layers.Dense(1, activation="relu", name=name + "_knn_mlp2")(knn_mlp1)

    # Compute attention coefficients.
    # Add point_cloud_mlp2 and point_cloud_mlp2 and apply softmax.
    attention_coefficients = layers.Lambda(lambda x: tf.add(
        tf.expand_dims(x[0], axis=-1),
        x[1]
    ), name=name + "_add")([pc_mlp2, knn_mlp2])
    attention_coefficients = layers.Permute([1, 3, 2])(attention_coefficients)
    attention_coefficients = layers.LeakyReLU()(attention_coefficients)
    attention_coefficients = layers.Activation("softmax", name=name + "_attention_coefficients")(attention_coefficients)

    # Compute attention features.
    # Matmul graph-features and attention coeeficients. Apply squeeze.
    attention_features = layers.Lambda(
        lambda x: tf.matmul(x[1], x[0])
    )([graph_features, attention_coefficients])
    attention_features = layers.Lambda(
        lambda x: K.squeeze(x, axis=2),
        name=name + "_attention_features"
    )(attention_features)

    return attention_features, graph_features, attention_coefficients
