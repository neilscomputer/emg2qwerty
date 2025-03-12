#!/bin/bash

PYTORCH_ENABLE_MPS_FALLBACK=1 HYDRA_FULL_ERROR=1 /usr/bin/time \
    python -m emg2qwerty.train user="glob(single_user)" trainer.devices=1 \
    name=tds-conv-lstm-spectogram-retry1 \
    model=tds_conv_lstm_ctc \
    trainer.max_epochs=150 --multirun

PYTORCH_ENABLE_MPS_FALLBACK=1 HYDRA_FULL_ERROR=1 /usr/bin/time \
    python -m emg2qwerty.train user="glob(single_user)" trainer.devices=1 \
    name=tds-conv-lstm-wavelet-no-log-retry1 \
    model=tds_conv_lstm_ctc \
    transforms=wavelet-no-log \
    trainer.max_epochs=150 --multirun

#PYTORCH_ENABLE_MPS_FALLBACK=1 HYDRA_FULL_ERROR=1 /usr/bin/time \
#    python -m emg2qwerty.train user="glob(single_user)" trainer.devices=1 \
#    name=banded-lstm \
#    model=banded_lstm \
#    trainer.max_epochs=100 --multirun
#
#PYTORCH_ENABLE_MPS_FALLBACK=1 HYDRA_FULL_ERROR=1 /usr/bin/time \
#    python -m emg2qwerty.train user="glob(single_user)" trainer.devices=1 \
#    name=conv-banded-lstm \
#    model=conv_banded_lstm \
#    trainer.max_epochs=100 --multirun

#PYTORCH_ENABLE_MPS_FALLBACK=1 HYDRA_FULL_ERROR=1 /usr/bin/time \
#    python -m emg2qwerty.train user="glob(single_user)" trainer.devices=1 \
#    name=ctc-lstm-test-1-wavelet-no-log \
#    model=tds_conv_lstm_ctc \
#    transforms=custom_transforms \
#    trainer.max_epochs=220 --multirun
