import torch.utils.data as data
import torch
import re
import os
import pandas as pd

class TatsuokaDataset(data.Dataset):
    def __init__(self, df):
        """Set the path for the CSV file containing TatsuokaDataset.
        Args:
            csv_path: path for CSV file for TatsuokaDataset
        """
	questions = df['question'].tolist()
        answers = df['correctanswer'].tolist()
        types = df['type'].tolist()

	idx = [3,5,6,7,9,11,12]
        search_a = r'\$(\d)?\s*(\\displaystyle )?(\{(\d+)\})?\s*(\\d?frac{(\d+)}{(\d+)})?\$'
	search_q = r'What is \$(\\displaystyle )?(\{(\d+)\})?\s*(\\d?frac{(\d+)}{(\d+)})?\s?([\-\+])\s?(\{(\d+)\})?(\\d?frac{(\d+)}{(\d+)})?\$\?'
	question_num = []
        answer_num = []
        filtered_types = []

	for i, (q, a, t) in enumerate(zip(questions, answers, types)):
            match_q = re.search(search_q, q)
            match_a = re.search(search_a, a)

            if check_20(match_q, match_a):
		question_num.append([match_q.group(i) for i in idx])
                filtered_types.append(t)
                if match_a.group(1):
                    answer_num.append([match_a.group(1), None,None]) # 11 means None
                elif match_a.group(4):
                    answer_num.append([match_a.group(4), match_a.group(6), match_a.group(7)])
                else:
                    answer_num.append([None, match_a.group(6), match_a.group(7)])
            else:
		# print "Something wrong",i, q, a
                pass
        
        self.questions = question_num
        self.answers = answer_num
        self.types = filtered_types

    def __getitem__(self, index):
        "returns one data pair, question and answers"
        q = self.questions[index]
        q_t = torch.cat( ( one_hot_21(q[0]),
                          one_hot_21(q[1]),
                          one_hot_21(q[2]),
                          one_hot_op(q[3]), 
                          one_hot_21(q[4]),
                          one_hot_21(q[5]),
                          one_hot_21(q[6])
                         ), 0)
        a = self.answers[index]
        a_t = torch.cat( ( one_hot_21(a[0]),
                         one_hot_21(a[1]),
                         one_hot_21(a[2])
                         ), 0)
        return q_t, a_t, self.types[index] 

    def __len__(self):
        return len(self.answers)

def check_20(q, a):
    idx_q = [3,5,6,9,11,12]
    for i in idx_q:
        if q.group(i) is not None and int(q.group(i)) > 20:
            return False
    idx_a = [4, 6, 7]
    for i in idx_a:
        if a.group(i) is not None and int(a.group(i)) > 20:
            return False
    return True
    
def decode(data):
    _, a1 = torch.max(data[:,0:21], 1)
    _, a2 = torch.max(data[:,21:42], 1)
    _, a3 = torch.max(data[:,42:63], 1)
    return torch.cat( ( decode_21(a1),
                         decode_21(a2),
                         decode_21(a3)
                         ), 1 )

def decode_21(idx):
    one_hot = torch.Tensor(idx.size()[0], 21)
    one_hot.zero_()
    idx = torch.unsqueeze(idx, 1)
    return one_hot.scatter_(1, idx, 1)
    
def one_hot_21(n_str):
    one_hot = torch.Tensor(21)
    one_hot.zero_()
    if n_str:
        n = int(n_str)
        one_hot.scatter_(0,torch.LongTensor([n]),1.0)
    else:
        one_hot.scatter_(0,torch.LongTensor([20]),1.0)
    return one_hot


def one_hot_op(op_str):
    one_hot = torch.Tensor(2)
    one_hot.zero_()
    if op_str == '+':
        one_hot.scatter_(0,torch.LongTensor([0]),1.0)
    else:
        one_hot.scatter_(0,torch.LongTensor([0]),1.0)
    return one_hot




def get_loader(csv_path, batch_size, shuffle, num_workers):
    df = pd.read_csv(csv_path)
    
    tat = TatsuokaDataset(df)
    
    # This will return (questions, answers) for every iteration.
    # questions: tensor of shape (batch_size, 128).
    # answers: tensor of shape (batch_size, 63).
    data_loader = torch.utils.data.DataLoader(dataset=tat, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader 


