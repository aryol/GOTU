import numpy as np
from utilities import generate_all_binaries    

# Number of samples set to zero indicates the generation of whole space. 
tasks = {
    # Task related to the f_1=x_0x_1+x_1x_2+x_2x_0 (Figures 1 & 7)
    'threesym': {
        'dimension': 15,
        'train_size': 0,
        'valid_size': 0,
        'test_size': 0,
        'target_function': lambda X: X[:, 0] * X[:, 1] - 1.25 * X[:, 1] * X[:, 2] + 1.5 * X[:, 2] * X[:, 0],
        'seen_condition': lambda X: ((X[:, 0] * X[:, 1] * X[:, 2] == 1)),
        'mask': np.array([[0,0,0], [0,0,1], [0,1,0], [1,0,0], [1,1,0], [0,1,1], [1,0,1], [1,1,1]]),
        'print_coefficients': True
    },


    # Task related to the function f_2=x_0x_1 in dimension 15 (Figures 1 & 2 & 6)
    '2parity': {
        'dimension': 15,
        'train_size': 0,
        'valid_size': 0,
        'test_size': 0,
        'target_function': lambda X: X[:, 0] * X[:, 1],
        'seen_condition': lambda X: ((X[:, 1] == 1) | (X[:, 0] == 1)),
        'mask': np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        'print_coefficients': True
    },


    # Task related to the f2=x_0x_1 embedded in dimension 40 (Figure 9)
    '2parity40': {
        'dimension': 40,
        'train_size': 2 ** 15,
        'valid_size': 2 ** 15,
        'test_size': 2 ** 15,
        'target_function': lambda X: X[:, 0] * X[:, 1],
        'seen_condition': lambda X: ((X[:, 1] == 1) | (X[:, 0] == 1)),
        'mask': np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        'print_coefficients': True
    },



    # Task related to the cyclic function f_3=x_0x_1x_2 + ... + x_14x_0x_1 (Figures 1 & 8)
    'cyclic3dim15': {
        'dimension': 15,
        'train_size': 0,
        'valid_size': 0,
        'test_size': 0,
        'target_function': lambda X: (X * np.roll(X, -1, axis=1) * np.roll(X, -2, axis=1)).sum(axis=1),
        'seen_condition': lambda X: ((X[:, 2] == 1) | (X[:, 1] == 1) | (X[:, 0] == 1)),
        'mask': np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]]),
        'print_coefficients': True
    }, 


    # Task related to the majority function embedded (f_4) in 40-dimensional space (Figure 10)
    'maj3dim40freeze2': {
        'dimension': 15,
        'train_size': 2 ** 15,
        'valid_size': 2 ** 15,
        'test_size': 2 ** 15,
        'target_function': lambda X: np.sign(X[:, :3].sum(axis=1)),
        'seen_condition': lambda X: ((X[:, 1] == 1) | (X[:, 0] == 1)),
        'mask': np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        'print_coefficients': True
    }, 


    # Task related to the experiment comparing RF models with ReLU activation and polynomial activation (Figure 5) 
    'rfexample': {
        'dimension': 18,
        'train_size': 0,
        'valid_size': 0,
        'test_size': 0,
        'target_function': lambda X: X[:, 0] * X[:, 1] * X[:, 2] + X[:, 0] * X[:, 3] * X[:, 4] * X[:, 5],
        'seen_condition': lambda X: ((X[:, 0] == 1)),
        'mask': np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                     [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                     [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]),
        'print_coefficients': True
    },

    # Tasks related to the length generalization experiment (Figure 3)
    'lengthgen15': {
        'dimension': 15,
        'train_size': 0,
        'valid_size': 0,
        'test_size': 0,
        'target_function': lambda X: X.prod(axis=1),
        'seen_condition': lambda X: (X.sum(axis=1) >= 15 - 2 * 15),
        'mask':  ((generate_all_binaries(15) + 1) / 2).astype(int),
        'print_coefficients': True
    },


    'lengthgen10': {
        'dimension': 15,
        'train_size': 0,
        'valid_size': 0,
        'test_size': 0,
        'target_function': lambda X: X.prod(axis=1),
        'seen_condition': lambda X: (X.sum(axis=1) >= 15 - 2 * 10),
        'mask':  ((generate_all_binaries(15) + 1) / 2).astype(int),
        'print_coefficients': True
    },


    'lengthgen9': {
        'dimension': 15,
        'train_size': 0,
        'valid_size': 0,
        'test_size': 0,
        'target_function': lambda X: X.prod(axis=1),
        'seen_condition': lambda X: (X.sum(axis=1) >= 15 - 2 * 9),
        'mask':  ((generate_all_binaries(15) + 1) / 2).astype(int),
        'print_coefficients': True
    },

    'lengthgen8': {
        'dimension': 15,
        'train_size': 0,
        'valid_size': 0,
        'test_size': 0,
        'target_function': lambda X: X.prod(axis=1),
        'seen_condition': lambda X: (X.sum(axis=1) >= 15 - 2 * 8),
        'mask':  ((generate_all_binaries(15) + 1) / 2).astype(int),
        'print_coefficients': True
    },

    'lengthgen7': {
        'dimension': 15,
        'train_size': 0,
        'valid_size': 0,
        'test_size': 0,
        'target_function': lambda X: X.prod(axis=1),
        'seen_condition': lambda X: (X.sum(axis=1) >= 15 - 2 * 7),
        'mask':  ((generate_all_binaries(15) + 1) / 2).astype(int),
        'print_coefficients': True
    },

    'lengthgen6': {
            'dimension': 15,
            'train_size': 0,
            'valid_size': 0,
            'test_size': 0,
            'target_function': lambda X: X.prod(axis=1),
            'seen_condition': lambda X: (X.sum(axis=1) >= 15 - 2 * 6),
            'mask':  ((generate_all_binaries(15) + 1) / 2).astype(int),
            'print_coefficients': True
        },


    # Tasks related to the curriculum learning experiment (Figure 4)
    'full16parity3000': {
        'dimension': 16,
        'train_size': 3000,
        'valid_size': 0,
        'test_size': 0,
        'target_function': lambda X: X.prod(axis=1),
        'seen_condition': lambda X: (True | (X[:, 0] == 1)),
        'mask': np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
        'print_coefficients': True
    },

    'full16parity5000': {
        'dimension': 16,
        'train_size': 5000,
        'valid_size': 0,
        'test_size': 0,
        'target_function': lambda X: X.prod(axis=1),
        'seen_condition': lambda X: (True | (X[:, 0] == 1)),
        'mask': np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
        'print_coefficients': True
    }, 


    'full16parity7000': {
        'dimension': 16,
        'train_size': 7000,
        'valid_size': 0,
        'test_size': 0,
        'target_function': lambda X: X.prod(axis=1),
        'seen_condition': lambda X: (True | (X[:, 0] == 1)),
        'mask': np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
        'print_coefficients': True
    }, 

    'full16parity9000': {
        'dimension': 16,
        'train_size': 9000,
        'valid_size': 0,
        'test_size': 0,
        'target_function': lambda X: X.prod(axis=1),
        'seen_condition': lambda X: (True | (X[:, 0] == 1)),
        'mask': np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
        'print_coefficients': True
    }, 
 

    'full16parity11000': {
        'dimension': 16,
        'train_size': 11000,
        'valid_size': 0,
        'test_size': 0,
        'target_function': lambda X: X.prod(axis=1),
        'seen_condition': lambda X: (True | (X[:, 0] == 1)),
        'mask': np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
        'print_coefficients': True
    },  

}