import unittest
import sys
sys.path.append("..")
from gapnet.layers import KNN, GraphAttention, MultiGraphAttention, Transform
from gapnet.model import build_model_paper
from tensorflow.keras import models, layers
import numpy as np
from tensorflow.keras.utils import plot_model

class TestMethods(unittest.TestCase):

    #@unittest.skip
    def test_knn(self):
        """
        Tests the KNN layer.
        """

        number_of_points = 1024
        features = 3
        k = 20

        # Create the model.
        model = models.Sequential()
        model.add(KNN(input_shape=(number_of_points, features), k=k))

        # Check if output is right.
        self.assertEqual(model.outputs[0].shape[1:], (number_of_points, k, features))

        # TODO Consider checking the KNN condition.

        # Do a prediction.
        input = np.array([(x , x, x) for x in range(number_of_points)])
        prediction = model.predict(np.expand_dims(input, axis=0))[0]
        print(prediction.shape)
        self.assertEqual(prediction.shape, (number_of_points, k, features))
        #print(prediction)
        #plt.imshow(prediction)
        #plt.show()
        #plt.close()

    #@unittest.skip
    def test_gap(self):
        """
        Tests the graph attention layer.
        """

        number_of_points = 1024
        features = 3
        k = 20
        features_out = 16

        # Input for a pointcloud.
        point_cloud = layers.Input(shape=(number_of_points, features))

        # Create the KNN layer and apply it to the input.
        knn = KNN(k=k, name="test_knn")(point_cloud)

        # Create the Graph Attention from point-cloud and KNN.
        graph_attention = GraphAttention(k=k, features=features_out)([point_cloud, knn])

        # Create the model.
        model = models.Model(point_cloud, graph_attention)

        # Check if output is right. The first is attention feature.
        self.assertEqual(model.outputs[0].shape[1:], (number_of_points, features_out))

        # Check if output is right. The second is graph feature.
        self.assertEqual(model.outputs[1].shape[1:], (number_of_points, k, features_out))

        # Check if output is right. The second is attention coefficients.
        self.assertEqual(model.outputs[2].shape[1:], (number_of_points, 1, k))

        # TODO Consider checking the KNN condition.

        # Do a prediction.
        input = np.array([(x , x, x) for x in range(number_of_points)])
        attention_features, graph_features, attention_coefficients = model.predict(np.expand_dims(input, axis=0))
        #print(prediction)
        #plt.imshow(prediction)
        #plt.show()
        #plt.close()


    #@unittest.skip
    def test_multigap_onehead(self):
        """
        Tests the multi head graph attention layer with one head.
        """

        number_of_points = 2048
        features = 3
        k = 20
        heads = 1
        features_out = 16

        # Input for a pointcloud.
        point_cloud = layers.Input(shape=(number_of_points, features))

        # Create the Graph Attention from point-cloud and KNN.
        multi_graph_attention = MultiGraphAttention(k=k, features=features_out, heads=heads)(point_cloud)

        # Create the model.
        model = models.Model(point_cloud, multi_graph_attention)

        # Check if output is right. The first is attention feature.
        self.assertEqual(model.outputs[0].shape[1:], (number_of_points, heads, features_out))

        # Check if output is right. The second is graph feature.
        self.assertEqual(model.outputs[1].shape[1:], (number_of_points, heads, k, features_out))

        # Check if output is right. The second is attention coefficients.
        self.assertEqual(model.outputs[2].shape[1:], (number_of_points, heads, k))


    #@unittest.skip
    def test_multigap_multihead(self):
        """
        Tests the multi head graph attention layer with multiple heads.
        """

        number_of_points = 1024
        features = 3
        k = 20
        heads = 4
        features_out = 16

        # Input for a pointcloud.
        point_cloud = layers.Input(shape=(number_of_points, features))

        # Create the Graph Attention from point-cloud and KNN.
        multi_graph_attention = MultiGraphAttention(k=k, features=features_out, heads=heads)(point_cloud)

        # Create the model.
        model = models.Model(point_cloud, multi_graph_attention)

        # Check if output is right. The first is attention feature.
        self.assertEqual(model.outputs[0].shape[1:], (number_of_points, heads, features_out))

        # Check if output is right. The second is graph feature.
        self.assertEqual(model.outputs[1].shape[1:], (number_of_points, heads, k, features_out))

        # Check if output is right. The second is attention coefficients.
        self.assertEqual(model.outputs[2].shape[1:], (number_of_points, heads, k))


    #@unittest.skip
    def test_transform(self):

        number_of_points = 1024
        features = 3
        k = 20
        features_out = 16

        # Input for a pointcloud.
        point_cloud = layers.Input(shape=(number_of_points, features))

        # Create the transform layer from point-cloud.
        point_cloud_transformed = Transform(k=k, features=features_out)(point_cloud)

        # Create the model.
        model = models.Model(point_cloud, point_cloud_transformed)

        # Check if output is right. The first is attention feature.
        self.assertEqual(model.outputs[0].shape[1:], (number_of_points, features))


    def test_model(self):
        # Shapes should be the same as in GAPNet CLS.
        point_cloud_input_shape = (1024, 3)
        attention_features_input_shape = (1024, 19)
        graph_features_input_shape = (1024, 20, 16)

        # Create the input.
        point_cloud_input = layers.Input(shape=point_cloud_input_shape)
        attention_features_input = layers.Input(shape=attention_features_input_shape)
        graph_features_input = layers.Input(shape=graph_features_input_shape)

        # Create the output.
        output = Transform(k=3)([point_cloud_input, attention_features_input, graph_features_input])

        # Create the model.
        model = models.Model([point_cloud_input, attention_features_input, graph_features_input], output)
        model.summary()

        # Test.
        number_of_samples = 3
        point_cloud_sample = np.random.random((number_of_samples,) + point_cloud_input_shape)
        attention_features_input = np.random.random((number_of_samples,) + attention_features_input_shape)
        graph_features_input = np.random.random((number_of_samples,) + graph_features_input_shape)
        preciction = model.predict([point_cloud_sample, attention_features_input, graph_features_input])
        print(prediction.shape)
        plt.imshow(prediction)


    #@unittest.skip
    def test_model(self):

        print("ARCHITECTURE")
        number_of_points = 1024
        features = 3
        nearest_neighbors = 20


        prediction_model, attention_model = build_model_paper(
            number_of_points,
            features,
            nearest_neighbors,
            heads1=1,
            heads2=4,
            bn_decay=None,
            output_size=1,
            output_activation="linear"
        )

        prediction_model.summary()
        plot_model(prediction_model, to_file='prediction_model.png')



        # See that the input shape is a pointcloud with 1024 points and three numbers each.
        self.assertEqual(prediction_model.get_layer("point_cloud_input").output_shape[0][1:], (1024, 3))

        # In the beginning one attention is used with just one head.
        self.assertEqual(prediction_model.get_layer("attention1_head_0_attention_features").output_shape[1:], (1024, 16))
        self.assertEqual(prediction_model.get_layer("attention1_head_0_graph_features").output_shape[1:], (1024, 20, 16))

        # The single head of the first attention goes big.
        self.assertEqual(prediction_model.get_layer("attention1_multi_attention_features").output_shape[1:], (1024, 1, 16))
        self.assertEqual(prediction_model.get_layer("attention1_multi_graph_features").output_shape[1:], (1024, 20, 1, 16))

        # TODO what is right shape of attention1_multi_attention_features_with_skip?


        # The first attention layer uses neighbors from KNN. Variable is called "neighbors" in original.
        #self.assertEqual(prediction_model.get_layer("attention1_all_local_features").output_shape[1:], (1024, 1, 19))

        #self.assertEqual(prediction_model.get_layer("attention1_all_global_features").output_shape[1:], (1024, 1, 1))

        #print("PREDICT")
        #input = np.random.random((number_of_points, 3))
        #input = np.array([(x, x, x) for x in range(number_of_points)])
        #print(input.shape)

        #prediction = prediction_model.predict(np.expand_dims(input, axis=0))[0]
        #print(prediction.shape)
        #print(prediction)
        #plt.imshow(prediction)
        #plt.show()
        #plt.close()



    # def test_upper(self):
    #     self.assertEqual('foo'.upper(), 'FOO')
    #
    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())
    #
    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

if __name__ == '__main__':
    unittest.main()
