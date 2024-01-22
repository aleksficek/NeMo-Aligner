# CAI (Constitutional AI)

## Environment and Setup
### Docker Container

### Data Sets

## Inference Service
### files:
1. [inference_service.py](inference_service.py) (replicated from [megatron_gpt_eval.py](https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/megatron_gpt_eval.py))
2. [inference_service_llama.yaml](conf/inference_service_llama.yaml) (replicated from [megatron_llama_inference.yaml](https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/conf/megatron_llama_inference.yaml)) 

### running inference service on a local machine
** tested on a local machine with  A6000 x 2

This is the script to run inference service on a local machine:
```batch
export CODE_NEMO_ALIGNER=<<path to NeMo-Aligner implementation folder>>
export MODEL_DIR=<<path to model folder, e.g., 'models/llama2'>>
export MODEL_FILE_NAME=<<model`s file name, e.g., 'llama2_7b_bf16.nemo'>>
export DOCKER_IMAGE_NAME=<<use latest docker container from nemo-aligner>>


#------------------------------------------------------------------------------
# prepare command to execute a python script with some arguments
#------------------------------------------------------------------------------
read -r -d '' cmd_run_inference_service <<EOF
export HYDRA_FULL_ERROR=1 \
&& export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
&& export PYTHONPATH="/workspace/code/NeMo-Aligner:${PYTHONPATH}" \
&& python -u /workspace/code/NeMo-Aligner/examples/nlp/cai/inference_service.py \
    --config-path=/workspace/code/NeMo-Aligner/examples/nlp/cai/conf \
    --config-name=inference_service_llama \
    trainer.num_nodes=1 \
    trainer.devices=2 \
    tensor_model_parallel_size=2 \
    pipeline_model_parallel_size=1 \
    gpt_model_file="/workspace/model/${MODEL_FILE_NAME}" \
	port=5656 \
	server=True
EOF


#------------------------------------------------------------------------------
# execute python script inside a container
#------------------------------------------------------------------------------
docker run \
      -v "${CODE_NEMO_ALIGNER}:/workspace/code/NeMo-Aligner" \
      -v "${MODEL_DIR}:/workspace/model" \
      --ipc=host \
      --net host \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      --gpus all \
      "${DOCKER_IMAGE_NAME}"  \
      bash -c "${cmd_run_inference_service}"
```

To send a request to the server, here is one example code:
```python
import json
import requests

batch_size = 1
port_num = 5656
headers = {"Content-Type": "application/json"}


def request_data(data):
    resp = requests.put('http://localhost:{}/generate'.format(port_num),
                        data=json.dumps(data),
                        headers=headers)
    sentences = resp.json()['sentences']
    return sentences


data = {
    "sentences": [""] * batch_size,
    "tokens_to_generate": 300,
    "temperature": 1.0,
    "add_BOS": True,
    "top_k": 0,
    "top_p": 0.9,
    "greedy": False,
    "all_probs": False,
    "repetition_penalty": 1.2,
    "min_tokens_to_generate": 2,
}

sentences = request_data(data)
```

### running inference service on a cluster
