import pickle
from sys import getsizeof

model_name = "./n-gram.pickle"
if __name__ == "__main__":
    with open(model_name, 'rb') as f:
        t_model = pickle.load(f)
    print(getsizeof(t_model))
    print(t_model.N())
    print('ngram count', t_model[['giáo', 'viên', 'trường']]['thpt'])
    print('ngram count', t_model[['nguyên', 'lý']]['của'])
    print('ngram count', t_model[['theo', 'tâm', 'lý']]['của'])
    print('count nguyen', t_model['nguyên'])
    print('count nguyen', t_model['cà'])