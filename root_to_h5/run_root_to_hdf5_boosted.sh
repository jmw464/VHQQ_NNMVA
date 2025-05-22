#!/bin/bash

input_dir=/global/cfs/cdirs/atlas/jmw464/run3_vhcc/mvatest/VHbb/
output_dir=/global/cfs/cdirs/atlas/jmw464/run3_vhcc/H5Training/kfold/

conda activate uproot-h5-env

python root_to_hdf5.py -i $input_dir -o $output_dir -m a -l 0
python root_to_hdf5.py -i $input_dir -o $output_dir -m d -l 0
python root_to_hdf5.py -i $input_dir -o $output_dir -m e -l 0
python root_to_hdf5.py -i $input_dir -o $output_dir -m a -l 1
python root_to_hdf5.py -i $input_dir -o $output_dir -m d -l 1
python root_to_hdf5.py -i $input_dir -o $output_dir -m e -l 1
python root_to_hdf5.py -i $input_dir -o $output_dir -m a -l 2
python root_to_hdf5.py -i $input_dir -o $output_dir -m d -l 2
python root_to_hdf5.py -i $input_dir -o $output_dir -m e -l 2