#!/bin/bash

# Completed
#PYTORCH_ENABLE_MPS_FALLBACK=1 HYDRA_FULL_ERROR=1 /usr/bin/time \
#    python -m emg2qwerty.train user="glob(single_user)" trainer.devices=1 \
#    name=baseline \
#    model=tds_conv_ctc \
#    trainer.max_epochs=150 --multirun

# Only half completed, need to re-run :(
PYTORCH_ENABLE_MPS_FALLBACK=1 HYDRA_FULL_ERROR=1 /usr/bin/time \
    python -m emg2qwerty.train user="glob(single_user)" trainer.devices=1 \
    name=baseline-wavelets-retry-1 \
    transforms=wavelet \
    model=tds_conv_ctc \
    trainer.max_epochs=150 --multirun

PYTORCH_ENABLE_MPS_FALLBACK=1 HYDRA_FULL_ERROR=1 /usr/bin/time \
    python -m emg2qwerty.train user="glob(single_user)" trainer.devices=1 \
    name=baseline-wavelets-no-log \
    transforms=wavelet-no-log \
    model=tds_conv_ctc \
    trainer.max_epochs=150 --multirun

#PYTORCH_ENABLE_MPS_FALLBACK=1 HYDRA_FULL_ERROR=1 /usr/bin/time \
#    python -m emg2qwerty.train user="glob(single_user)" trainer.devices=1 \
#    name=ctc-lstm-test-1-wavelet-no-log \
#    model=tds_conv_lstm_ctc \
#    transforms=custom_transforms \
#    trainer.max_epochs=220 --multirun
