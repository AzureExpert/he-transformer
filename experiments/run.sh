#!/bin/bash

LD_LIBRARY_PATH=$HOME/repo/venvs/he3/lib/python3.5/site-packages/ngraph NGRAPH_TF_BACKEND=HE_HEAAN NGRAPH_HE_HEAAN_CONFIG=heaan_config.json python gemm_timing.py --out foo.txt
