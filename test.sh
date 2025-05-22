#!/bin/bash
INPUT_DIR=test-input
OUTPUT_DIR=test-output
WORKDIR=test-workdir
CONFIG_FILE=configs/config.json

python process.py --input_path $INPUT_DIR/Task101_Example_sl_bin_clf-fold0 --output_path $OUTPUT_DIR/Task101_Example_sl_bin_clf-fold0 --workdir $WORKDIR/Task101_Example_sl_bin_clf-fold0 --input_split_dir $WORKDIR/Task101_Example_sl_bin_clf-fold0/input_split --config_file $CONFIG_FILE
python process.py --input_path $INPUT_DIR/Task104_Example_ml_bin_clf-fold0 --output_path $OUTPUT_DIR/Task104_Example_ml_bin_clf-fold0 --workdir $WORKDIR/Task104_Example_ml_bin_clf-fold0 --input_split_dir $WORKDIR/Task104_Example_ml_bin_clf-fold0/input_split --config_file $CONFIG_FILE
python process.py --input_path $INPUT_DIR/Task107_Example_ml_reg-fold0 --output_path $OUTPUT_DIR/Task107_Example_ml_reg-fold0 --workdir $WORKDIR/Task107_Example_ml_reg-fold0 --input_split_dir $WORKDIR/Task107_Example_ml_reg-fold0/input_split --config_file $CONFIG_FILE
python process.py --input_path $INPUT_DIR/Task108_Example_sl_ner-fold0 --output_path $OUTPUT_DIR/Task108_Example_sl_ner-fold0 --workdir $WORKDIR/Task108_Example_sl_ner-fold0 --input_split_dir $WORKDIR/Task108_Example_sl_ner-fold0/input_split --config_file $CONFIG_FILE