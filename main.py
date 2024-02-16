import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer
BERT_PATH = './model/chinese_wwm_L-12_H-768_A-12/publish'
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
labels = {'1': 0,
          '2': 1,
          '3': 2,
          '4': 3,
          '5': 4
          }

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels[str(label)] for label in df['rating']]
        self.texts = [tokenizer(comment,
                                padding='max_length',
                                max_length=512,
                                truncation=True,
                                return_tensors="pt")
                      for comment in df['comments']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


if __name__ == '__main__':
    # 导入并划分数据集
    np.random.seed(112)
    df = pd.read_csv('./data/train.csv')[:10000]
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8 * len(df)), int(.9 * len(df))])

    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    print(len(df_train), len(df_val), len(df_test))

    # 查看数据集信息
    # print(df.head())
    # print(df_train.iloc[17197])
    # print(df['rating'].unique())
    # print(df.info(verbose=True, show_counts=True))

    # 训练模型train model
    from creat_model import BertClassifier
    from train import train

    EPOCHS = 5
    model = BertClassifier(BERT_PATH=BERT_PATH)
    LR = 1e-6
    train(model, df_train, df_val, LR, EPOCHS)

    # 评估模型
    from evaluate import evaluate
    evaluate(model, df_test)


