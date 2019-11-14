import unittest
import sys
sys.path.append("..")
from gapnet.layers import KNN, GraphAttention, MultiGraphAttention, Transform
from gapnet.models import GAPNet
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
        graph_attention = GraphAttention(features_out)([point_cloud, knn])

        # Create the model.
        model = models.Model(point_cloud, graph_attention)
        model.summary()

        # Check if output is right. The first is attention feature.
        self.assertEqual(model.outputs[0].shape[1:], (number_of_points, 1, features_out))

        # Check if output is right. The second is graph feature.
        self.assertEqual(model.outputs[1].shape[1:], (number_of_points, 1, k, features_out))

        # Check if output is right. The second is attention coefficients.
        self.assertEqual(model.outputs[2].shape[1:], (number_of_points, 1, k))

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

        number_of_points = 1024
        features_in = 3
        k = 20
        heads = 1
        features_out = 16

        # Input for a pointcloud.
        point_cloud = layers.Input(shape=(number_of_points, features_in))

        # Create the Graph Attention from point-cloud and KNN.
        multi_graph_attention = MultiGraphAttention(k=k, features_out=features_out, heads=heads)(point_cloud)

        # Create the model.
        model = models.Model(point_cloud, multi_graph_attention)

        # Check if output is right. The first is attention feature.
        self.assertEqual(model.outputs[0].shape[1:], (number_of_points, heads, features_out))

        # Check if output is right. The second is graph feature.
        self.assertEqual(model.outputs[1].shape[1:], (number_of_points, k, features_out))

        # Check if output is right. The second is attention coefficients.
        self.assertEqual(model.outputs[2].shape[1:], (number_of_points, heads, k))


    #@unittest.skip
    def test_multigap_multihead(self):
        """
        Tests the multi head graph attention layer with multiple heads.
        """

        number_of_points = 1024
        features_in = 3
        k = 20
        heads = 4
        features_out = 16

        # Input for a pointcloud.
        point_cloud = layers.Input(shape=(number_of_points, features_in))

        # Create the Graph Attention from point-cloud and KNN.
        multi_graph_attention = MultiGraphAttention(k=k, features_out=features_out, heads=heads)(point_cloud)

        # Create the model.
        model = models.Model(point_cloud, multi_graph_attention)

        # Check if output is right. The first is attention feature.
        self.assertEqual(model.outputs[0].shape[1:], (number_of_points, 1, heads * features_out))

        # Check if output is right. The second is graph feature.
        self.assertEqual(model.outputs[1].shape[1:], (number_of_points, k, features_out * heads))

        # Check if output is right. The second is attention coefficients.
        self.assertEqual(model.outputs[2].shape[1:], (number_of_points, 1, heads * k))


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

        # Create the model.
        model = GAPNet()
        model.summary()

        for x in model.non_trainable_weights:
            if "normalization" not in str(x):
                print(x)


if __name__ == '__main__':
    unittest.main()
