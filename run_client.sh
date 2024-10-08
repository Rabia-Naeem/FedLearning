
python training_fl.py --server 127.0.0.1:8080 --seed $RANDOM  --folder data --early-stop 10 --normalisation ln --num-clients 8 --partition $1 --diffusion --diffusion-folder "run_nonIID_fl_protein_8" &
#  export TF_ENABLE_ONEDNN_OPTS=0