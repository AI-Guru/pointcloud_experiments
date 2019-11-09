import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from data_loader import DataGenerator
from schedules import onetenth_50_75
import os
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import sys
from model_cls import create_pointnet
from gapnet.model import create_gapnet_dev

model_names = ["pointnet", "gapnet_dev"]


def main():

    # Check command line arguments.
    if len(sys.argv) != 2 or sys.argv[1] not in model_names:
        print("Must provide name of model.")
        print("Options: " + " ".join(model_names))
        exit(0)
    model_name = sys.argv[1]

    # Data preparation.
    nb_classes = 40
    train_file = './ModelNet40/ply_data_train.h5'
    test_file = './ModelNet40/ply_data_test.h5'

    # Hyperparameters.
    number_of_points = 1024
    epochs = 100
    batch_size = 32

    # Data generators for training and validation.
    train = DataGenerator(train_file, batch_size, number_of_points, nb_classes, train=True)
    val = DataGenerator(test_file, batch_size, number_of_points, nb_classes, train=False)

    # Create the model.
    if model_name == "pointnet":
        model = create_pointnet(number_of_points, nb_classes)
    elif model_name == "gapnet_dev":
        model = create_gapnet_dev(number_of_points, nb_classes)
    model.summary()

    # Ensure output paths.
    if not os.path.exists('./results/'):
        os.mkdir('./results/')
    if not os.path.exists('./results/{}'.format(model_name)):
        os.mkdir('./results/{}'.format(model_name))


    # Compile the model.
    lr = 0.0001
    adam = Adam(lr=lr)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    checkpoint = ModelCheckpoint('./results/{}/model.h5'.format(model_name), monitor='val_acc',
                                 save_weights_only=True, save_best_only=True,
                                 verbose=1)

    # Create the callbacks.
    callbacks = []

    # Logging training progress with tensorboard.
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir="logs/{}".format(model_name),
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=True,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        embeddings_data=None,
        update_freq="epoch"
    )
    callbacks.append(tensorboard_callback)

    # Train the model.
    history = model.fit_generator(train.generator(),
                                  steps_per_epoch=9840 // batch_size,
                                  epochs=epochs,
                                  validation_data=val.generator(),
                                  validation_steps=2468 // batch_size,
                                  callbacks=[checkpoint, onetenth_50_75(lr), tensorboard_callback],
                                  verbose=1)

    plot_history(history, './results/{}'.format(model_name))
    save_history(history, './results/{}'.format(model_name))
    model.save_weights('./results/{}/model_weights.h5'.format(model_name))


def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()


if __name__ == '__main__':
    main()
