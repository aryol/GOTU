# !/bin/bash

for seed in {1..10}
do
    # Function $f_1$ as in Figures 1 & 7
    python3 main.py -seed $seed -task threesym -model transformer -lr 0.00002 -epochs 40 -batch-size 256 -opt adam
    python3 main.py -seed $seed -task threesym -model rfrelu -lr 0.00001 -epochs 40 -batch-size 256
    python3 main.py -seed $seed -task threesym -model mlp -lr 0.003 -epochs 40 -batch-size 64 
    python3 main.py -seed $seed -task threesym -model meanfield -lr 1000.0 -epochs 40 -batch-size 64
    
    # Function $f_2$ as in Figures 1 & 2
    python3 main.py -seed $seed -task 2parity -model mlp -lr 0.003 -epochs 50 -batch-size 64
    python3 main.py -seed $seed -task 2parity -model meanfield -lr 1000.0 -epochs 50 -batch-size 64
    python3 main.py -seed $seed -task 2parity -model rfrelu -lr 0.00001 -epochs 40 -batch-size 256
    python3 main.py -seed $seed -task 2parity -model transformer -lr 0.00001 -epochs 20 -batch-size 256 -opt adam

    # Function $f_3$ as in Figures 1 & 8
    python3 main.py -seed $seed -task cyclic3dim15 -model rfrelu -lr 0.0003 -epochs 50 -batch-size 256
    python3 main.py -seed $seed -task cyclic3dim15 -model mlp -lr 0.003 -epochs 100 -batch-size 64 
    python3 main.py -seed $seed -task cyclic3dim15 -model transformer -lr 0.00002 -epochs 80 -batch-size 256 -opt adam
    python3 main.py -seed $seed -task cyclic3dim15 -model meanfield -lr 1000.0 -epochs 100 -batch-size 64

    # Function $f_4$ as in Figure 10
    python3 main.py -seed $seed -task maj3dim40freeze2 -model mlp -lr 0.003 -epochs 40 -batch-size 64
    python3 main.py -seed $seed -task maj3dim40freeze2 -model transformer -lr 0.00001 -epochs 30 -batch-size 256 -opt adam
    python3 main.py -seed $seed -task maj3dim40freeze2 -model meanfield -lr 1000.0 -epochs 40 -batch-size 64
    python3 main.py -seed $seed -task maj3dim40freeze2 -model rfrelu -lr 0.0001 -epochs 80 -batch-size 256


    # Length generalization experiment of Figure 3
    python3 main.py -seed $seed -task lengthgen6 -model mlp -lr 0.00001 -epochs 100 -batch-size 64 -opt adam -compute-int 10
    python3 main.py -seed $seed -task lengthgen7 -model mlp -lr 0.00001 -epochs 100 -batch-size 64 -opt adam -compute-int 10
    python3 main.py -seed $seed -task lengthgen8 -model mlp -lr 0.00001 -epochs 100 -batch-size 64 -opt adam -compute-int 10
    python3 main.py -seed $seed -task lengthgen9 -model mlp -lr 0.00001 -epochs 100 -batch-size 64 -opt adam -compute-int 10
    python3 main.py -seed $seed -task lengthgen10 -model mlp -lr 0.00001 -epochs 100 -batch-size 64 -opt adam -compute-int 10

    # Function f_2 in ambient dim=40 of Figure 9
    python3 main.py -seed $seed -task 2parity40 -model mlp -lr 0.003 -epochs 50 -batch-size 64
    python3 main.py -seed $seed -task 2parity40 -model meanfield -lr 1000.0 -epochs 50 -batch-size 64
    python3 main.py -seed $seed -task 2parity40 -model rfrelu -lr 0.00001 -epochs 100 -batch-size 256
    python3 main.py -seed $seed -task 2parity40 -model transformer -lr 0.00001 -epochs 20 -batch-size 256 -opt adam


    # Curriculum experiment in Figure 4
    ## Normal training
    python3 main.py -seed $seed -task full16parity3000 -model mlp -lr 0.0001 -epochs 2000 -batch-size 64 -opt adam -compute-int 5 -curr-step 16
    python3 main.py -seed $seed -task full16parity5000 -model mlp -lr 0.0001 -epochs 2000 -batch-size 64 -opt adam -compute-int 5 -curr-step 16
    python3 main.py -seed $seed -task full16parity7000 -model mlp -lr 0.0001 -epochs 2000 -batch-size 64 -opt adam -compute-int 5 -curr-step 16
    python3 main.py -seed $seed -task full16parity9000 -model mlp -lr 0.0001 -epochs 2000 -batch-size 64 -opt adam -compute-int 5 -curr-step 16
    python3 main.py -seed $seed -task full16parity11000 -model mlp -lr 0.0001 -epochs 2000 -batch-size 64 -opt adam -compute-int 5 -curr-step 16
    ## Degree-curriculum
    python3 main.py -seed $seed -task full16parity3000 -model mlp -lr 0.0001 -epochs 2000 -batch-size 64 -opt adam -compute-int 5 -curr-step 4
    python3 main.py -seed $seed -task full16parity5000 -model mlp -lr 0.0001 -epochs 2000 -batch-size 64 -opt adam -compute-int 5 -curr-step 4
    python3 main.py -seed $seed -task full16parity7000 -model mlp -lr 0.0001 -epochs 2000 -batch-size 64 -opt adam -compute-int 5 -curr-step 4
    python3 main.py -seed $seed -task full16parity9000 -model mlp -lr 0.0001 -epochs 2000 -batch-size 64 -opt adam -compute-int 5 -curr-step 4
    python3 main.py -seed $seed -task full16parity11000 -model mlp -lr 0.0001 -epochs 2000 -batch-size 64 -opt adam -compute-int 5 -curr-step 4

    # Random features ReLU vs poly experiment in Figure 5
    python3 main.py -seed $seed -task rfexample -model rfrelu -lr 0.0003 -epochs 100 -batch-size 256 
    python3 main.py -seed $seed -task rfexample -model rfpoly -lr 0.0003 -epochs 100 -batch-size 256


    # Learning rate sensitivity result in Figure 6
    python3 main.py -seed $seed -task 2parity -model mlp -lr 0.15 -epochs 4 -batch-size 64
    python3 main.py -seed $seed -task 2parity -model mlp -lr 0.1 -epochs 4 -batch-size 64
    python3 main.py -seed $seed -task 2parity -model mlp -lr 0.03 -epochs 40 -batch-size 64
    python3 main.py -seed $seed -task 2parity -model mlp -lr 0.01 -epochs 50 -batch-size 64 
    python3 main.py -seed $seed -task 2parity -model mlp -lr 0.003 -epochs 50 -batch-size 64
    python3 main.py -seed $seed -task 2parity -model mlp -lr 0.001 -epochs 50 -batch-size 64 -compute-int 20
    python3 main.py -seed $seed -task 2parity -model mlp -lr 0.0003 -epochs 60 -batch-size 64 -compute-int 20
    python3 main.py -seed $seed -task 2parity -model mlp -lr 0.0001 -epochs 100 -batch-size 64 -compute-int 20
    python3 main.py -seed $seed -task 2parity -model mlp -lr 0.00003 -epochs 250 -batch-size 64 -compute-int 20

done