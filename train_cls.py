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
import shutil

model_names = ["pointnet", "gapnet_dev"]

model_name = "gapnet_dev"
training_name = "03"


def main():

    # Check command line arguments.
    #if len(sys.argv) != 2 or sys.argv[1] not in model_names:
    #    print("Must provide name of model.")
    #    print("Options: " + " ".join(model_names))
    #    exit(0)
    #model_name = sys.argv[1]

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
    output_path = "logs"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = os.path.join(output_path, model_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = os.path.join(output_path, training_name)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)


    # Compile the model.
    lr = 0.0001
    adam = Adam(lr=lr)
    model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )

    # Checkpoint callback.
    checkpoint = ModelCheckpoint(
        os.path.join(output_path, "model.h5"),
        monitor="val_acc",
        save_weights_only=True,
        save_best_only=True,
        verbose=1
        )

    # Logging training progress with tensorboard.
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=output_path,
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

    # Train the model.
    history = model.fit_generator(
        train.generator(),
        #steps_per_epoch=9840 // batch_size,
        steps_per_epoch=1,
        #epochs=epochs,
        epochs=2,
        validation_data=val.generator(),
        #validation_steps=2468 // batch_size,
        validation_steps=2,
        callbacks=[checkpoint, onetenth_50_75(lr), tensorboard_callback],
        verbose=1
        )

    # Save history and model.
    plot_history(history, output_path)
    save_history(history, output_path)
    model.save_weights(os.path.join(output_path, "model_weights.h5"))


def plot_history(history, result_dir):
    if "acc" in history.history:
        plt.plot(history.history['acc'], marker='.')
        plt.plot(history.history['val_acc'], marker='.')
    elif "accuracy" in history.history:
        plt.plot(history.history['accuracy'], marker='.')
        plt.plot(history.history['val_accuracy'], marker='.')
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
    if "acc" in history.history:
        acc = history.history['acc']
        val_acc = history.history['val_acc']
    elif "accuracy" in history.history:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()


if __name__ == '__main__':
    main()
