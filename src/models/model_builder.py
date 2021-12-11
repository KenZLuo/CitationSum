import copy
import dgl
import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import GraphConv
from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer
from fastNLP.core import seq_len_to_mask

def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if(large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, mask):
        if(self.finetune):
            if x.shape[1] > 512:
                encoder_outputs = []
                h_cnode_batch = []
                n = x.shape[1] // 512
                for w in range(n+1):
                    last_hidden, pool_out = self.model(input_ids=x[: , w*512: w*512 + 512],
                            attention_mask= mask[: , w*512: w*512 + 512])
                    encoder_outputs.append(last_hidden)
                    h_cnode_batch.append(pool_output)
                    encoder_outputs = torch.cat(encoder_outputs, dim=1)
                    h_cnode_batch = torch.max(torch.stack(h_cnode_batch, dim=0), dim=0)[0].squeeze(0)
            else:
                #print(mask)
                encoder_outputs, h_cnode_batch = self.model(input_ids=x,  attention_mask=mask)
                #encoder_outputs = outputs.last_hidden_state
                #h_cnode_batch = outputs.pooler_output
        else:
            self.eval()
            with torch.no_grad():
                if x.shape[1] > 512:
                    encoder_outputs = []
                    h_cnode_batch = []
                    n = x.shape[1] // 512
                    for w in range(n+1):
                        last_hidden, pool_output = self.model(input_ids=x[: , w*512: w*512 + 512],
                                attention_mask=mask[: , w*512: w*512 + 512])
                        encoder_outputs.append(last_hidden)
                        h_cnode_batch.append(pool_output)
                        encoder_outputs = torch.cat(encoder_outputs, dim=1)
                        h_cnode_batch = torch.max(torch.stack(h_cnode_batch, dim=0), dim=0)[0].squeeze(0)

                else:
                    encoder_outputs, h_cnode_batch = self.model(x, attention_mask=mask)
                    #encoder_outputs = outputs.last_hidden_state
                    #h_cnode_batch = outputs.pooler_output

        return encoder_outputs, h_cnode_batch

class GNNEncoder(nn.Module):
    def __init__(self, args):
        super(GNNEncoder, self).__init__()
        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        if args.GNN == "GAT":
            self.gnn = GAT(num_layers=args.num_layers, in_dim=args.hidden_dim , heads=heads,
                           num_hidden=args.hidden_dim, residual=args.residual)
        elif args.GNN == "GCN":
            self.gnn = GCN(in_feats=args.hidden_dim, n_hidden=args.hidden_dim ,
                           n_layers=args.num_layers, dropout=args.gnn_drop)
        else:
            raise Exception("GNN not supported ")

    def forward(self, graphs, node_feats, node_idx, nodes_num_batch):
        # if graphs length = 1 there will be errors in dgl
        if len(graphs) == 1:
            graphs.append(dgl.DGLGraph())

        g = dgl.batch(graphs)
        if g.number_of_nodes() != len(node_feats):
            logger.error("error: number of nodes in dgl graph do not equal nodes in input graph !!!")
            logger.error(
                f"number of nodes this batch:{sum(nodes_num_batch).item()}, number of num in dgl graph {g.number_of_nodes()}")
            assert g.number_of_nodes() == len(node_feats)
        g = g.to(node_feats.device)
        gnn_feat = self.gnn(g, node_feats)
        b = len(nodes_num_batch)
        n = max(nodes_num_batch)
        h = gnn_feat.shape[1]
        node_features = torch.zeros([b, n, h], device=gnn_feat.device)
        # 还原成 B x max_nodes_num x hidden
        for i in range(len(node_idx) - 1):
            curr_idx = node_idx[i]
            next_idx = node_idx[i + 1]
            mask = torch.arange(curr_idx, next_idx, device=gnn_feat.device)
            output_feat = torch.index_select(gnn_feat, 0, mask)
            if output_feat.shape[0] < n:
                pad_num = n - output_feat.shape[0]
                extra_zeros = torch.zeros(pad_num, h, device=gnn_feat.device)
                output_feat = torch.cat([output_feat, extra_zeros], 0)
            node_features[i] = output_feat

        return node_features

class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 heads,
                 num_hidden=256,
                 activation=F.elu,
                 feat_drop=0.1,
                 attn_drop=0.0,
                 negative_slope=0.2,
                 residual=True,
                 out_dim=None):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers - 1):
            h = self.gat_layers[l](g, h).flatten(1)
        # output layer mean of the attention head
        output = self.gat_layers[-1](g, h).mean(1)
        return output


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_layers,
                 activation=F.relu,
                 dropout=0.1,
                 out_dim=None):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h) + h  # residual
        return h


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)
        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
                                     num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)
        self.gnnEncoder = GNNEncoder(args)
        self.join = nn.Linear(2 * self.bert.model.config.hidden_size, self.bert.model.config.hidden_size)
        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def pooling(self, pooler_output, last_hidden_state):
        if self.args.pooling == "none":
            node_feature = pooler_output
        elif self.args.pooling == "mean":
            node_feature = torch.mean(last_hidden_state, dim=1)
        elif self.args.pooling == "max":
            node_feature = torch.max(last_hidden_state, dim=1)[0]
        else:
            mean_feat = torch.mean(last_hidden_state, dim=1)
            max_feat = torch.max(last_hidden_state, dim=1)[0]
            feat = torch.cat([mean_feat, max_feat], 1)
            #print (feat.shape, self.join)
            node_feature = self.join(feat)

        return node_feature.unsqueeze(1)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls, graph_src, graph, graph_len, node_num):
        encoder_outputs, h_cnode_batch = self.bert(src, mask_src)

        node_features = [self.pooling(h_cnode_batch, encoder_outputs)]
        neighbor_node_num = max(node_num - 1)

        for idx in range(neighbor_node_num):
            #print(idx)
            #print(graph_src)
            #print(graph_len)
            graph_src = torch.tensor(graph_src)
            node_batch = graph_src[:, idx, :]
            len_batch =torch.tensor(graph_len)[:, idx].clone()
            # there may be some error if  seq_len = 0 in this batch
            for i in range(len(len_batch)):
                len_batch[i] += (len_batch[i] == 0)
            node_enc_mask = seq_len_to_mask(len_batch, max_len=self.args.max_graph_pos)
            node_enc_outputs, node_hidden = self.bert(node_batch.to(src.device), node_enc_mask.to(src.device))
            node_features.append(self.pooling(node_hidden, node_enc_outputs))
        node_features = torch.cat(node_features, 1)

        node_feature_res = []
        node_feature_idx = [0]
        for idx, node_feature in enumerate(node_features):
            n_num = node_num[idx]
            mask = torch.arange(n_num)
            node_feature_idx.append(node_feature_idx[-1] + len(mask))
            node_feature_res.append(torch.index_select(node_feature, 0, torch.tensor(mask, device=node_feature.device)))
        node_feature_res = torch.cat(node_feature_res, 0)
        assert len(node_feature_res) == sum(node_num).item()

        neighbor_feat = self.gnnEncoder(graph, node_feature_res, node_feature_idx, node_num)
        #print(neighbor_feat.shape)

        dec_state = self.decoder.init_decoder_state(src, encoder_outputs)
        decoder_outputs, state = self.decoder(tgt[:, :-1], encoder_outputs, dec_state)
        return decoder_outputs, None
