A Federated Learning model for Drug-Target Interaction.

Running Steps:

Run on powershell

1. python create_data.py data
2. python dataset_nonIID_split.py n data,
   (where n represents the number of splits you want to create which would be equal to the number of clients.)

Run on Git Bash

export TF_ENABLE_ONEDNN_OPTS=0 
3. bash run_server.sh
for every client on a new terminal:
export TF_ENABLE_ONEDNN_OPTS=0 
4. bash run_client.sh n,
(where n represents the number of client you are executing. it would go something like:
bash run_client.sh 0
bash run_client.sh 1
bash run_client.sh 2
.
.
.
bash run_client n)
