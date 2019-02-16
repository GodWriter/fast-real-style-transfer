import os

from solver import Solver
from dataloader import DataSet
from config import parse_args

args = parse_args()
data = DataSet(args)
solver = Solver(args)

if __name__ == '__main__':
    module = args.module

    # select the GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

    if module == 'test_dataset':
        data.test_dataset()
    elif module == 'create_dataset':
        data.create_dataset()
    elif module == 'train':
        solver.train()
    elif module == 'test':
        solver.test()
