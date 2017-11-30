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
    train_data_loader = get_loader(args.csv_path, args.batch_size,
			    shuffle=True, num_workers=args.num_workers)

    net = Net()
    # Loss and Optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate )
    total_step = len(train_data_loader)
    
    for epoch in range(args.num_epochs):
	running_loss=0.0
	for i, data in enumerate(train_data_loader):
            questions, answers, types = data
            questions, answers = Variable(questions), Variable(answers)
            optimizer.zero_grad()
            outputs = net(questions)
            loss = criterion(outputs, answers)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % args.log_step == 0:
		print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
			%(epoch, args.num_epochs, i, total_step, 
			loss.data[0], np.exp(loss.data[0]))) 

    print "Saving Final model"
    torch.save(net.state_dict(), 
               os.path.join(args.model_path, 
                            'model.pkl' ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/fc/' ,
                        help='path for saving trained models')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=50,
                        help='step size for saving trained models')
    parser.add_argument('--csv_path', type=str, default='./data/train.csv', 
                        help='path for Tatsuoka CSV file')
    # Model parameters
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)


