A Federated Learning model for Drug-Target Interaction.

Running Steps:
1. export export TF_ENABLE_ONEDNN_OPTS=0
2. python create_data.py data
3. python dataset_nonIID_split.py data n, 
    (where n represents the number of splits you want to create which would be equal to the number of clients.)
4. bash run_server.sh
5. bash run_client.sh n, 
    (where n represents the number of client you are executing. it would go something like:
    bash run_client.sh 0
    bash run_client.sh 1
    bash run_client.sh 2
    .
    .
    .
    bash run_client n)
    
