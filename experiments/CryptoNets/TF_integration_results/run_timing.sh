NGRAPH_TF_VLOG_LEVEL=5 NGRAPH_BATCHED_TENSOR=1 NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json NGRAPH_TF_BACKEND=HE:SEAL:CKKS python test.py --batch_size=1
NGRAPH_TF_VLOG_LEVEL=5 NGRAPH_BATCHED_TENSOR=1 NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_14.json NGRAPH_TF_BACKEND=HE:SEAL:CKKS python test.py --batch_size=1

