
python training_fl.py --server localhost:8080 --seed $RANDOM  --folder data --early-stop 50 --normalisation ln --num-clients 4 --partition $1 --diffusion --diffusion-folder "run_nonIID_fl_protein_4" &
#  export TF_ENABLE_ONEDNN_OPTS=0