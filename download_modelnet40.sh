#!/usr/bin/sh

wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
unzip modelnet40_ply_hdf5_2048.zip
mkdir ModelNet40
mkdir ModelNet40/test
mkdir ModelNet40/train
cp modelnet40_ply_hdf5_2048/ply_data_train*.h5 ModelNet40/train/
cp modelnet40_ply_hdf5_2048/ply_data_test*.h5 ModelNet40/test/
rm modelnet40_ply_hdf5_2048.zip
rm -rf modelnet40_ply_hdf5_2048
