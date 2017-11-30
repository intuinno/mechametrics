import argparse
import torch
from data_loader import get_loader, decode
from model import Net
from torch.autograd import Variable
import pickle
import os
import numpy as np

def main(args):
    if not os.path.exists(args.model_path):
	os.makedirs(args.model_path)
    test_data_loader = get_loader(args.csv_path, args.batch_size,
			    shuffle=True, num_workers=args.num_workers)

    net = Net()
    # Loss and Optimizer
    
    net.load_state_dict(torch.load(args.model_path))

    correct = 0
    total = 0
    types_val = [9037002, 9123201, 9123301, 9123401, 9123501, 9123601, 9123701,
                       9123801, 9123901, 9124001, 9124101, 9124201, 9124301, 9124401,
                       9124501]
    accuracy = {}
    for t in types_val:
        accuracy[t] = { 'correct': 0, 'total': 0 }
    for i, data in enumerate(test_data_loader):
	questions, answers, types = data
	outputs = net(Variable(questions))
        prediction = decode(outputs.data) 
        for i in range(answers.size()[0]):
            total += 1
            accuracy[types[i]]['total'] += 1
            if all(torch.eq(prediction[i], answers[i])):
                accuracy[types[i]]['correct'] += 1
                correct +=1

    print accuracy
    print "total accuracy %f"%(100.0*correct/total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/fc/model.pkl' ,
                        help='path for loading trained models')
    parser.add_argument('--csv_path', type=str, default='./data/test.csv', 
                        help='path for Tatsuoka CSV file')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    print(args)
    main(args)


