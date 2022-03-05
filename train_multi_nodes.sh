#!/bin/sh
num_machines=2
machine_rank=$1
num_gpus=4
master_ip='10.72.77.31'
master_port=21162
config_file="configs/ablation_studies/pointset/unsup_pvt_m.yaml"
output_dir="output/pvt_m_2x_adamw"

echo $machine_rank
echo ${master_ip}
echo $config_file

GLOO_SOCKET_IFNAME=bond0 python train_net.py --num-machines ${num_machines}\
 --machine-rank ${machine_rank} --num-gpus ${num_gpus} --dist-url tcp://${master_ip}:${master_port}\
 --config-file ${config_file} OUTPUT_DIR ${output_dir}