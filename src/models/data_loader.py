import bisect
import gc
import glob
import random

import torch
import numpy as np
from others.logging import logger



class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def _pad_graph_inputs(self, graph_inputs, pad_id, max_len):
        #print(max_len)
        pad_graph_inputs = []
        graph_input_len = []
        node_num = []
        max_node_num = max([len(graph_input) for graph_input in graph_inputs])
        if max_node_num ==0:
            return torch.empty(len(graph_inputs),1),torch.zeros(len(graph_inputs),1),torch.ones(len(graph_inputs),1)
        # Here we pad each graph with max node number and pad each src with max length. Note we put positive
        # sample and negative samples of the same node together.
        # Notice that node num and token num is not the same in each graph
        # rtn_data max_node_num x sample_num x max_token_num
        # each_input_len node_num x sample_num x 1

        for graph_input in graph_inputs:
            e_rtn_data = []
            e_input_len = []
            for i in range(max_node_num):
                if i < len(graph_input):
                    e_rtn_data.append(graph_input[i] + [pad_id] * (max_len - len(graph_input)))
                    e_input_len.append(len(graph_input[i]))
                else:
                    #print(graph_input)
                    e_rtn_data.append([pad_id] *max_len)
                    e_input_len.append(0)
            #rtn_data = [g + [pad_id] * (max_len - len(g)) for g in graph_input]
            pad_graph_inputs.append(e_rtn_data)
            graph_input_len.append(e_input_len)
            node_num.append(len(graph_input)+1)
        #print(pad_graph_inputs)
        #print(graph_input_len)
        return torch.tensor(pad_graph_inputs), torch.tensor(graph_input_len), torch.tensor(node_num)

    def _pad_graph(self, graph_inputs):
        #print(max_len)
        pad_graph_inputs = []
        graph_input_len = []
        node_num = []
        max_node_num = max([graph_input.shape(0) for graph_input in graph_inputs])
        graph = []
        for graph_input in graph_inputs:
            pad_arr_row = np.zeros(max_node_num-graph_input.shape(0), graph_input.shape(0))
            each_graph = np.concatenate((graph_input, pad_arr_row), axis=0)
            pad_arr_col = np.zeros(max_node_num - graph_input.shape(1), graph_input.shape(0))
            each_graph = np.concatenate((each_graph, pad_arr_col), axis=1)
            graph.append(each_graph)
        return graph

    def __init__(self, args, data=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            pre_src = [x[0] for x in data]
            pre_tgt = [x[1] for x in data]
            pre_segs = [x[2] for x in data]
            pre_clss = [x[3] for x in data]
            pre_src_sent_labels = [x[4] for x in data]
            pre_graph_src = [x[5] for x in data]
            pre_neg_graph_src = [x[6] for x in data]
            pre_graph = [x[7] for x in data]

            src = torch.tensor(self._pad(pre_src, 0))
            tgt = torch.tensor(self._pad(pre_tgt, 0))
            graph_src, graph_src_len, node_num = self._pad_graph_inputs(pre_graph_src, 0, args.max_graph_pos)
            neg_graph_src, neg_graph_src_len, neg_node_num = self._pad_graph_inputs(pre_neg_graph_src, 0, args.max_graph_pos)
            graph = torch.tensor(self._pad_graph(pre_graph))
           # print(len(graph_src))
            segs = torch.tensor(self._pad(pre_segs, 0))
            mask_src = ~(src == 0)
            mask_tgt = ~(tgt == 0)
            #print(mask_src)

            clss = torch.tensor(self._pad(pre_clss, -1))
            src_sent_labels = torch.tensor(self._pad(pre_src_sent_labels, 0))
            mask_cls = ~(clss == -1)
            clss[clss == -1] = 0
            setattr(self, 'clss', clss.to(device))
            setattr(self, 'mask_cls', mask_cls.to(device))
            setattr(self, 'src_sent_labels', src_sent_labels.to(device))


            setattr(self, 'src', src.to(device))
            setattr(self, 'tgt', tgt.to(device))
            setattr(self, 'segs', segs.to(device))
            setattr(self, 'mask_src', mask_src.to(device))
            #setattr(self, 'graph_src', mask_tgt.to(device))
            setattr(self, 'graph_src', graph_src.to(device))
            setattr(self, 'neg_graph_src', neg_graph_src.to(device))
            setattr(self, 'mask_tgt', mask_tgt.to(device))
            setattr(self, 'graph_src_len', graph_src_len.to(device))
            setattr(self, 'neg_graph_src_len', neg_graph_src_len.to(device))
            setattr(self, 'graph', graph.to(device))
            setattr(self, 'node_num', node_num.to(device))
            setattr(self, 'neg_node_num', neg_node_num.to(device))
            if (is_test):
                src_str = [x[-2] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[-1] for x in data]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size




def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset
    #print(corpus_type)
    #print(args.bert_data_path + corpus_type + '.0.pt')
    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.bert_data_path + corpus_type + '.[0-9]*.pt'))

    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.bert_data_path + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def abs_batch_size_fn(new, count):
    src, tgt = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents=0
        max_n_tokens=0
    max_n_sents = max(max_n_sents, len(tgt))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    if (count > 6):
        return src_elements + 1e3
    return src_elements


def ext_batch_size_fn(new, count):
    if (len(new) == 4):
        pass
    src, labels = new[0], new[4]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets,  batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)


    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args = self.args,
            dataset=self.cur_dataset,  batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset,  batch_size, device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0
        if (self.args.task == 'abs'):
            self.batch_size_fn = abs_batch_size_fn
        else:
            self.batch_size_fn = ext_batch_size_fn

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        src = ex['src']
        tgt = ex['tgt'][:self.args.max_tgt_len][:-1]+[2]
        src_sent_labels = ex['src_sent_labels']
        segs = ex['segs']
        if(not self.args.use_interval):
            segs=[0]*len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']
        graph_src = ex['graph_src']
        neg_graph_src = ex['neg_graph_src']
        graph = ex['graph']
        end_id = [src[-1]]
        src = src[:-1][:self.args.max_pos - 1] + end_id

        segs = segs[:self.args.max_pos]
        #print(len(graph_src))
        #print(graph_src)
        graph_src_max = []
        for s in graph_src:
            #new_s = []
            #new_s.append(s[0]+s[1]+s[2])
            #new_s.append(s[3])
            #new_s.append(s[4])
            if s == []:
                each_src = s
            else:
                each_src = [each_s[:-1][:self.args.max_graph_pos - 1] + end_id for each_s in s]
            graph_src_max.append(each_src)

        neg_graph_src_max = []
        for s in neg_graph_src:
            # new_s = []
            # new_s.append(s[0]+s[1]+s[2])
            # new_s.append(s[3])
            # new_s.append(s[4])
            if s == []:
                each_src = s
            else:
                each_src = [each_s[:-1][:self.args.max_graph_pos - 1] + end_id for each_s in s]
            neg_graph_src_max.append(each_src)

        #graph_src = [s[:-1][:self.args.max_graph_pos - 1] + end_id for s in graph_src]
        max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
        src_sent_labels = src_sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]
        # src_txt = src_txt[:max_sent_id]

        if(is_test):
            return src, tgt, segs, clss, src_sent_labels, graph_src_max, neg_graph_src_max, graph, src_txt, tgt_txt
        else:
            return src, tgt, segs, clss, src_sent_labels, graph_src_max, neg_graph_src_max, graph

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex['src'])==0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if(ex is None):
                continue
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):

            if (self.args.task == 'abs'):
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = sorted(p_batch, key=lambda x: len(x[1]))
            else:
                p_batch = sorted(buffer, key=lambda x: len(x[2]))

            p_batch = self.batch(p_batch, self.batch_size)


            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if(len(b)==0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(self.args, minibatch, self.device, self.is_test)

                yield batch
            return


class TextDataloader(object):
    def __init__(self, args, datasets, batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.batch_size = batch_size
        self.device = device

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        src = ex['src']
        tgt = ex['tgt'][:self.args.max_tgt_len][:-1] + [2]
        src_sent_labels = ex['src_sent_labels']
        segs = ex['segs']
        if (not self.args.use_interval):
            segs = [0] * len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']

        end_id = [src[-1]]
        src = src[:-1][:self.args.max_pos - 1] + end_id
        segs = segs[:self.args.max_pos]
        max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
        src_sent_labels = src_sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]
        # src_txt = src_txt[:max_sent_id]

        if (is_test):
            return src, tgt, segs, clss, src_sent_labels, src_txt, tgt_txt
        else:
            return src, tgt, segs, clss, src_sent_labels

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if (len(ex['src']) == 0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if (ex is None):
                continue
            minibatch.append(ex)
            size_so_far = simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):
            if (self.args.task == 'abs'):
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = sorted(p_batch, key=lambda x: len(x[1]))
            else:
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = batch(p_batch, self.batch_size)

            p_batch = batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if (len(b) == 0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return
