'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask


TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SST_CLASSES = 5
N_CFIMDB_CLASSES = 2
N_QUORA_CLASSES = 2
N_STS_CLASSES = 6

class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### 
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sst_classifier = nn.Linear(config.hidden_size, N_SST_CLASSES)
        self.cfimdb_classifier = nn.Linear(config.hidden_size, N_CFIMDB_CLASSES)
        self.quora_classifier = nn.Linear(2 * config.hidden_size, 1)
        self.sts_classifier = nn.Linear(2 * config.hidden_size, 1)

        # raise NotImplementedError


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### 

        outputs = self.bert.forward(input_ids, attention_mask)
        first_tk = outputs['pooler_output']

        # output = self.dropout(first_tk)
        # output = self.classifier(output)

        output = first_tk

        return output
        raise NotImplementedError


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### 

        output = self.forward(input_ids, attention_mask)
        output = self.dropout(output)
        output = self.sst_classifier(output)

        return output
        raise NotImplementedError


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### 

        output1 = self.forward(input_ids_1, attention_mask_1)
        output2 = self.forward(input_ids_2, attention_mask_2)

        # dropout before
        output1 = self.dropout(output1)
        output2 = self.dropout(output2)

        # concatenated = torch.cat((output1, output2), dim=1) # concatenate with no separator token
        concatenated = torch.cat((output1, output2), dim=-1).squeeze(1)

        # output = self.dropout(concatenated)
        output = self.quora_classifier(concatenated)

        return output
    
        raise NotImplementedError


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### 

        output1 = self.forward(input_ids_1, attention_mask_1)
        output2 = self.forward(input_ids_2, attention_mask_2)

        output1 = self.dropout(output1)
        output2 = self.dropout(output2)

        concatenated = torch.cat((output1, output2), dim=-1).squeeze(1) # concatenate with no separator token

        output = self.dropout(concatenated)
        output = self.sts_classifier(output)

        return output

        raise NotImplementedError



def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    # random.shuffle(para_train_data)
    para_train_data_small = para_train_data[:755*32]
    # random.shuffle(para_dev_data)
    # para_dev_data = para_train_data[:108*8]
    para_dev_data = para_dev_data[:200*32]

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    para_train_data_small = SentencePairDataset(para_train_data_small, args)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=8,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=8,
                                    collate_fn=sst_dev_data.collate_fn)
    
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=32,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=32,
                                    collate_fn=para_dev_data.collate_fn)
    
    para_train_data_small_loader = DataLoader(para_train_data_small, shuffle=True, batch_size=32,
                                      collate_fn=para_train_data_small.collate_fn)
    
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=8,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=8,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    model_eval_multitask(sst_dev_dataloader, para_dev_dataloader,
                                                   sts_dev_dataloader, model, device)
    # model_eval_multitask(sst_train_dataloader, para_train_data_small_loader, 
    #                                                    sts_train_dataloader, model, device)
    
    print("TRAINING ON SST")

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        cur_loss = 0
        for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_sentiment(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            cur_loss += loss.item()

            if(num_batches % 100 == 0):
                print("Batches " + str(num_batches) + " loss " + str(cur_loss / 100))
                cur_loss = 0

        train_loss = train_loss / (num_batches)

        # train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        # dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        # train_acc, train_f1, *_ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, 
        #                                                sts_train_dataloader, model, device)
        (sentiment_accuracy,sst_y_pred, sst_sent_ids,
                paraphrase_accuracy, para_y_pred, para_sent_ids,
                sts_corr, sts_y_pred, sts_sent_ids) = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader,
                                                   sts_dev_dataloader, model, device)
        
        dev_acc = 1/3*sentiment_accuracy + 1/3*paraphrase_accuracy + 1/6*(sts_corr+1)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        # print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")

    print("TRAINING ON QUORA")
    count = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        cur_loss = 0
        for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                       batch['attention_mask_1'], batch['token_ids_2'], 
                                       batch['attention_mask_2'], batch['labels'])

            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)

            # loss = F.cross_entropy(logits.to(torch.float), b_labels.view(logits.shape).to(torch.float), reduction='sum') / 32
            # loss = F.l1_loss(logits.sigmoid().to(torch.float), b_labels.view(logits.shape).to(torch.float)) / args.batch_size
            # print(logits.flatten().shape)
            # print(b_labels.flatten().shape)
            loss = F.binary_cross_entropy_with_logits(logits.flatten().float().cuda(), b_labels.flatten().float().cuda()) / 32
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            cur_loss += loss.item()

            if(num_batches % 100 == 0):
                print("Batches " + str(num_batches) + " loss " + str(cur_loss / 100))
                cur_loss = 0

            if(num_batches == 800):
                break

        train_loss = train_loss / (num_batches)

        # train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        # dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        (sentiment_accuracy,sst_y_pred, sst_sent_ids,
                paraphrase_accuracy, para_y_pred, para_sent_ids,
                sts_corr, sts_y_pred, sts_sent_ids) = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader,
                                                   sts_dev_dataloader, model, device)
        
        dev_acc = 1/3*sentiment_accuracy + 1/3*paraphrase_accuracy + 1/6*(sts_corr+1)
        

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        # print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")

    print("TRAINING ON STS")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        cur_loss = 0
        for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                       batch['attention_mask_1'], batch['token_ids_2'], 
                                       batch['attention_mask_2'], batch['labels'])

            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
            # loss = F.cross_entropy(logits.to(torch.float), b_labels.view(logits.shape).to(torch.float), reduction='sum') / args.batch_size
            # loss = F.l1_loss(logits.sigmoid().to(torch.float), b_labels.view(logits.shape).to(torch.float)) / args.batch_size
            loss = F.mse_loss(logits.flatten().float().cuda(), b_labels.flatten().float().cuda()) / 8

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            cur_loss += loss.item()

            if(num_batches % 100 == 0):
                print("Batches " + str(num_batches) + " loss " + str(cur_loss / 100))
                cur_loss = 0

        train_loss = train_loss / (num_batches)

        # train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        # dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        # train_acc, train_f1, *_ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, 
        #                                                sts_train_dataloader, model, device)
        (sentiment_accuracy,sst_y_pred, sst_sent_ids,
                paraphrase_accuracy, para_y_pred, para_sent_ids,
                sts_corr, sts_y_pred, sts_sent_ids) = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader,
                                                   sts_dev_dataloader, model, device)
        
        dev_acc = 1/3*sentiment_accuracy + 1/3*paraphrase_accuracy + 1/6*(sts_corr+1)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        # print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")

    save_model(model, optimizer, args, config, "sequential_05_21_final.pt")

def train_quora(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    # random.shuffle(para_train_data)
    para_train_data_small = para_train_data[:755*32]
    # random.shuffle(para_dev_data)
    # para_dev_data = para_train_data[:108*8]
    para_dev_data = para_dev_data[:200*32]

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    para_train_data_small = SentencePairDataset(para_train_data_small, args)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=8,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=8,
                                    collate_fn=sst_dev_data.collate_fn)
    
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=32,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=32,
                                    collate_fn=para_dev_data.collate_fn)
    
    para_train_data_small_loader = DataLoader(para_train_data_small, shuffle=True, batch_size=32,
                                      collate_fn=para_train_data_small.collate_fn)
    
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=8,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=8,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model.
    # config = {'hidden_dropout_prob': args.hidden_dropout_prob,
    #           'num_labels': num_labels,
    #           'hidden_size': 768,
    #           'data_dir': '.',
    #           'fine_tune_mode': args.fine_tune_mode}

    # config = SimpleNamespace(**config)

    # model = MultitaskBERT(config)
    # model = model.to(device)

    saved = torch.load('full-model-10-1e-05-1e-5-linear-lr-decay-best-05-27.pt')
    config = saved['model_config']

    model = MultitaskBERT(config)
    model.load_state_dict(saved['model'])
    model = model.to(device)

    lr = args.lr
    # optimizer = AdamW(model.parameters(), lr=lr)

    optimizer = AdamW(
        [
            {"params": model.quora_classifier.parameters(), "lr": 1e-3 },
        ], lr=0)
    
    best_dev_acc = 0

    model_eval_multitask(sst_dev_dataloader, para_dev_dataloader,
                                                   sts_dev_dataloader, model, device)
    # model_eval_multitask(sst_train_dataloader, para_train_data_small_loader, 
    #                                                    sts_train_dataloader, model, device)

    print("FINE-TUNING QUORA LAYER")
    count = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        cur_loss = 0
        for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                       batch['attention_mask_1'], batch['token_ids_2'], 
                                       batch['attention_mask_2'], batch['labels'])

            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)

            loss = F.binary_cross_entropy_with_logits(logits.flatten().float().cuda(), b_labels.flatten().float().cuda()) / 32
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            cur_loss += loss.item()

            if(num_batches % 100 == 0):
                print("Batches " + str(num_batches) + " loss " + str(cur_loss / 100))
                cur_loss = 0

            if(num_batches == 800):
                break

        train_loss = train_loss / (num_batches)

        (sentiment_accuracy,sst_y_pred, sst_sent_ids,
                paraphrase_accuracy, para_y_pred, para_sent_ids,
                sts_corr, sts_y_pred, sts_sent_ids) = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader,
                                                   sts_dev_dataloader, model, device)
        
        dev_acc = 1/3*sentiment_accuracy + 1/3*paraphrase_accuracy + 1/6*(sts_corr+1)
        

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, "paraphrase_ft_from_05-27_05_22_best.pt")

        # print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")

    save_model(model, optimizer, args, config, "paraphrase_ft_from_05-27_05_22_final.pt")

def train_sts(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    # random.shuffle(para_train_data)
    para_train_data_small = para_train_data[:755*32]
    # random.shuffle(para_dev_data)
    # para_dev_data = para_train_data[:108*8]
    para_dev_data = para_dev_data[:200*32]

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    para_train_data_small = SentencePairDataset(para_train_data_small, args)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=8,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=8,
                                    collate_fn=sst_dev_data.collate_fn)
    
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=32,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=32,
                                    collate_fn=para_dev_data.collate_fn)
    
    para_train_data_small_loader = DataLoader(para_train_data_small, shuffle=True, batch_size=32,
                                      collate_fn=para_train_data_small.collate_fn)
    
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=8,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=8,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model.
    # config = {'hidden_dropout_prob': args.hidden_dropout_prob,
    #           'num_labels': num_labels,
    #           'hidden_size': 768,
    #           'data_dir': '.',
    #           'fine_tune_mode': args.fine_tune_mode}

    # config = SimpleNamespace(**config)

    # model = MultitaskBERT(config)
    # model = model.to(device)

    saved = torch.load('full-model-10-1e-05-1e-5-linear-lr-decay-best-05-27.pt')
    config = saved['model_config']

    model = MultitaskBERT(config)
    model.load_state_dict(saved['model'])
    model = model.to(device)

    lr = args.lr
    # optimizer = AdamW(model.parameters(), lr=lr)

    optimizer = AdamW(
        [
            {"params": model.sts_classifier.parameters(), "lr": 1e-3 },
        ], lr=0)
    
    best_dev_acc = 0

    model_eval_multitask(sst_dev_dataloader, para_dev_dataloader,
                                                   sts_dev_dataloader, model, device)
    # model_eval_multitask(sst_train_dataloader, para_train_data_small_loader, 
    #                                                    sts_train_dataloader, model, device)

    print("FINE-TUNING STS LAYER")
    count = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        cur_loss = 0
        for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                       batch['attention_mask_1'], batch['token_ids_2'], 
                                       batch['attention_mask_2'], batch['labels'])

            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)

            loss = F.mse_loss(logits.flatten().float().cuda(), b_labels.flatten().float().cuda()) / 8
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            cur_loss += loss.item()

            if(num_batches % 100 == 0):
                print("Batches " + str(num_batches) + " loss " + str(cur_loss / 100))
                cur_loss = 0

            if(num_batches == 800):
                break

        train_loss = train_loss / (num_batches)

        (sentiment_accuracy,sst_y_pred, sst_sent_ids,
                paraphrase_accuracy, para_y_pred, para_sent_ids,
                sts_corr, sts_y_pred, sts_sent_ids) = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader,
                                                   sts_dev_dataloader, model, device)
        
        dev_acc = 1/3*sentiment_accuracy + 1/3*paraphrase_accuracy + 1/6*(sts_corr+1)
        

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, "sst_ft_from_05-27_05_28_best.pt")

        # print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")

    save_model(model, optimizer, args, config, "paraphrase_ft_from_05-27_05_28_final.pt")

def train_multitask_epoch_round_robin(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    # random.shuffle(para_train_data)
    para_train_data_small = para_train_data[:100*32] # 755
    random.shuffle(para_dev_data)
    para_dev_data = para_dev_data[:200*32]

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    para_train_data_small = SentencePairDataset(para_train_data_small, args)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=8,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=8,
                                    collate_fn=sst_dev_data.collate_fn)
    
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=32,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=32,
                                    collate_fn=para_dev_data.collate_fn)
    
    para_train_data_small_loader = DataLoader(para_train_data_small, shuffle=True, batch_size=32,
                                      collate_fn=para_train_data_small.collate_fn)
    
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=8,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=8,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    model_eval_multitask(sst_dev_dataloader, para_dev_dataloader,
                                                   sts_dev_dataloader, model, device)
    # model_eval_multitask(sst_train_dataloader, para_train_data_small_loader, 
    #                                                    sts_train_dataloader, model, device)
    
    
    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        loss1 = 0
        loss2 = 0
        loss3 = 0
        num_batches = 0
        print("TRAINING ON SST")
        cur_loss = 0
        for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_sentiment(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / 8

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loss1 += loss.item()

            cur_loss += loss.item()

            if(num_batches % 100 == 0):
                print("Batches " + str(num_batches) + " loss " + str(cur_loss / 100))
                cur_loss = 0

            num_batches += 1

        loss1 /= num_batches

        (sentiment_accuracy,sst_y_pred, sst_sent_ids,
                paraphrase_accuracy, para_y_pred, para_sent_ids,
                sts_corr, sts_y_pred, sts_sent_ids) = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader,
                                                   sts_dev_dataloader, model, device)
        
        print("TRAINING ON QUORA")
        model.train()
        cur_loss = 0
        cur_batches = 0
        for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                       batch['attention_mask_1'], batch['token_ids_2'], 
                                       batch['attention_mask_2'], batch['labels'])

            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)

            # loss = F.cross_entropy(logits.to(torch.float), b_labels.view(logits.shape).to(torch.float), reduction='sum') / args.batch_size
            # loss = F.l1_loss(logits.sigmoid().to(torch.float), b_labels.view(logits.shape).to(torch.float)) / 32
            loss = F.binary_cross_entropy_with_logits(logits.flatten().float().cuda(), b_labels.flatten().float().cuda()) / 32
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            cur_loss += loss.item()
            loss2 += loss.item()

            if(cur_batches % 100 == 0):
                print("Batches " + str(cur_batches) + " loss " + str(cur_loss / 100))
                cur_loss = 0

            if(cur_batches >= 800):
                break
            num_batches += 1
            cur_batches += 1

        loss2 /= cur_batches

        (sentiment_accuracy,sst_y_pred, sst_sent_ids,
                paraphrase_accuracy, para_y_pred, para_sent_ids,
                sts_corr, sts_y_pred, sts_sent_ids) = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader,
                                                   sts_dev_dataloader, model, device)
        
        print("TRAINING ON STS")
        model.train()
        cur_loss = 0
        cur_batches = 0
        for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                       batch['attention_mask_1'], batch['token_ids_2'], 
                                       batch['attention_mask_2'], batch['labels'])

            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
            loss = F.mse_loss(logits.flatten().float().cuda(), b_labels.flatten().float().cuda()) / 8

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loss3 += loss.item()
            num_batches += 1
            cur_batches += 1

            cur_loss += loss.item()

            if(cur_batches % 100 == 0):
                print("Batches " + str(cur_batches) + " loss " + str(cur_loss / 100))
                cur_loss = 0

        train_loss = train_loss / (num_batches)
        loss3 /= cur_batches

        (sentiment_accuracy,sst_y_pred, sst_sent_ids,
                paraphrase_accuracy, para_y_pred, para_sent_ids,
                sts_corr, sts_y_pred, sts_sent_ids) = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader,
                                                   sts_dev_dataloader, model, device)
        
        dev_acc = 1/3*sentiment_accuracy + 1/3*paraphrase_accuracy + 1/6*(sts_corr+1)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        # print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")
        print(f"sst loss :: {loss1 :.3f}, para loss :: {loss2 :.3f}, sts loss : {loss3 :.3f}")

    save_model(model, optimizer, args, config, "epoch_round_robin_05_21_final.pt")


def train_multitask_round_robin(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    # random.shuffle(para_train_data)
    random.shuffle(para_dev_data)
    para_dev_data = para_dev_data[:200*32]

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=8,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=8,
                                    collate_fn=sst_dev_data.collate_fn)
    
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=32,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=32,
                                    collate_fn=para_dev_data.collate_fn)
    
    
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=8,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=8,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    model_eval_multitask(sst_dev_dataloader, para_dev_dataloader,
                                                   sts_dev_dataloader, model, device)
    
    print("TRAINING ROUND ROBIN")

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        loss1 = 0
        loss2 = 0 
        loss3 = 0
        prev1 = 0
        prev2 = 0
        prev3 = 0

        for batch_temp in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            num_batches += 1

            for batch in sst_train_dataloader:

                b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                loss1 += loss.item()
                break

            for i in range(1):
                for batch in para_train_dataloader:
                    b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                        batch['attention_mask_1'], batch['token_ids_2'], 
                                        batch['attention_mask_2'], batch['labels'])

                    b_ids_1 = b_ids_1.to(device)
                    b_mask_1 = b_mask_1.to(device)
                    b_ids_2 = b_ids_2.to(device)
                    b_mask_2 = b_mask_2.to(device)
                    b_labels = b_labels.to(device)

                    optimizer.zero_grad()
                    logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)

                    loss = F.binary_cross_entropy_with_logits(logits.flatten().float().cuda(), b_labels.flatten().float().cuda()) / 32

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    loss2 += loss.item()
                    break

            for batch in sts_train_dataloader:

                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                       batch['attention_mask_1'], batch['token_ids_2'], 
                                       batch['attention_mask_2'], batch['labels'])

                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)

                loss = F.mse_loss(logits.flatten().float().cuda(), b_labels.flatten().float().cuda()) / 8

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                loss3 += loss.item()
                break

            if(num_batches % 50 == 0):
                print(num_batches, loss1-prev1, loss2-prev2, loss3-prev3, train_loss-prev1-prev2-prev3, train_loss)
                prev1 = loss1
                prev2 = loss2
                prev3 = loss3 

        train_loss = train_loss / (3*num_batches)
        loss1 /= num_batches
        loss2 /= (1*num_batches)
        loss3 /= num_batches

        dev_acc, dev_f1, *_ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader,
                                                   sts_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")
        print(f"sst loss :: {loss1 :.3f}, para loss :: {loss2 :.3f}, sts loss : {loss3 :.3f}")

    save_model(model, optimizer, args, config, "batch_round_robin_05_21_final.pt")

    

def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-paraphrase-best-05-22.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    # train_multitask(args)
    train_sts(args)
    # train_multitask_round_robin(args)
    # train_multitask_epoch_round_robin(args)
    test_multitask(args)