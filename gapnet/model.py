import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras import backend as K
from .layers import KNN, Transform, MultiGraphAttention, GraphAttention


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
