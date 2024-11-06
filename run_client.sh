
#!/bin/bash

# Set world size and rank based on the partition index (client number)
world_size=5
rank=$(( $1 + 1 ))

# Run client with specified rank in a new background process
python training_fl.py --server localhost:8080 --seed $RANDOM --folder data --early-stop 50 \
--normalisation ln --num-clients 4 --partition $1 --diffusion --diffusion-folder "run_nonIID_fl_protein_4" \
--rank $rank --world-size $world_size &


#  export TF_ENABLE_ONEDNN_OPTS=0