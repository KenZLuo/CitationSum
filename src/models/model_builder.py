import copy
import dgl
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import GraphConv
from models.decoder_with_graph import TransformerDecoderWithGraph
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
                    h_cnode_batch.append(pool_out)
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
            #logger.error("error: number of nodes in dgl graph do not equal nodes in input graph !!!")
            print(
                f"number of nodes this batch:{sum(nodes_num_batch).item()}, number of num in dgl graph {g.number_of_nodes()} node feats {len(node_feats)}")

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
            nodes_src = torch.ones([b, n], device=gnn_feat.device)
            if output_feat.shape[0] < n:
                pad_num = n - output_feat.shape[0]
                extra_zeros = torch.zeros(pad_num, h, device=gnn_feat.device)
                output_feat = torch.cat([output_feat, extra_zeros], 0)
                nodes_src[i, n-pad_num: ] -= 1
            node_features[i] = output_feat
            
        return node_features, nodes_src

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
        # output layer mean of the attention 
        #print(self.gat_layers)
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
        self.join = nn.Linear(2 * self.bert.model.config.hidden_size, self.args.hidden_dim)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.loss = nn.CrossEntropyLoss(reduction='none')
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

        self.decoder_with_graph = TransformerDecoderWithGraph(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)
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

    def forward(self, src, tgt, mask_src, graph_src, graph, graph_len, node_num):
        encoder_outputs, h_cnode_batch = self.bert(src, mask_src)
        #print(src.shape, mask_src.shape)
        node_features = [self.pooling(h_cnode_batch, encoder_outputs)]
        #node_features = [hidden_outputs]
        # node_num = B x 1
        neighbor_node_num = max(node_num - 1)

        if neighbor_node_num == 0:
            # There is no nodes with neighbors
            # print ("Neighbor node zero calculation")
            dec_state = self.decoder.init_decoder_state(src, encoder_outputs)
            decoder_outputs, state = self.decoder(tgt[:, :-1], encoder_outputs, dec_state)
            return decoder_outputs, None, None, None

        # graph_src = B x node_num x negative_sample+1 x max_len
        # graph_len = B x node_num x negative_sample+1
        all_features = []
        batch_size, nn, negative_num, max_len = graph_src.size()
        graph_batch = graph_src.reshape(-1, graph_src.size(-1))
        len_batch = (graph_len.reshape(-1) == 0)
        graph_enc_mask = seq_len_to_mask(len_batch, max_len=self.args.max_graph_pos)
        graph_enc_outputs, graph_hidden = self.bert(graph_batch, graph_enc_mask)
        graph_features = self.pooling(graph_hidden, graph_enc_outputs)
        graph_features = graph_features.reshape(nn, negative_num, batch_size, -1)
        self_features = node_features[0].permute(1, 0, 2).unsqueeze(1).repeat(1, negative_num, 1, 1)
        print (self_features.shape, graph_features.shape, batch_size, nn, negative_num, max_len)
        graph_features = torch.cat([self_features, graph_features], dim=0)
        graph_features = graph_features.reshape((nn+1)*negative_num, batch_size, -1)
        # (node_num x negative_num) x batch_size x hidden_size
        graph_features = graph_features.permute(1, 0, 2)
        norm_graph_features = graph_features / (graph_features.norm(dim=-1)[:, :, None])
        norm_encoder_outputs = encoder_outputs / (encoder_outputs.norm(dim=-1)[:, :, None])
        doc_word_cos_sim = torch.bmm(norm_encoder_outputs, norm_graph_features.permute(0, 2, 1))
        # labels = torch.zeros(encoder_outputs.size(1)).long().to(doc_word_cos_sim.device)
        # doc_word_contra_loss = self.loss(doc_word_cos_sim.squeeze(0), labels).mean()
        # print (f"Doc word contra loss is {doc_word_contra_loss}")
        # batch_size x word_num x (node_num x negative_num)
        # for i in range(self.args.negative_number+1):
        #     temp_features = node_features
        #     for idx in range(neighbor_node_num):
        #         node_batch = torch.squeeze(graph_src[:, idx,i,:],1)
        #         len_batch =graph_len[:, idx,i].clone()
        #         for j in range(len(len_batch)):
        #             len_batch[j] += (len_batch[j] == 0)
        #             node_enc_mask = seq_len_to_mask(len_batch, max_len=self.args.max_graph_pos)
        #             node_enc_outputs, node_hidden = self.bert(node_batch, node_enc_mask)
        #             temp_features.append(self.pooling(node_hidden, node_enc_outputs))
        #     temp_feat = torch.cat(temp_features, 1)
        #     all_features.append(temp_feat) # batch_size * node_num * hidden_size

        # pos_sim = self.cos(encoder_outputs, all_features[0][:, 0, :]).unsqueeze(2)
        # all_sim = [pos_sim]
        # for i in range(neighbor_node_num):
        #     each_sim = self.cos(encoder_outputs, all_features[1][:, i+1, :]).unsqueeze(2)
        #     all_sim.append(each_sim)
        # doc_word_sim_logits = torch.cat(all_sim, 2) / self.args.temp

        # doc_word_labels = torch.zeros(doc_word_sim_logits.shape[0], doc_word_sim_logits.shape[1],
        #                               dtype=torch.int64).to(encoder_outputs.device)

        # doc_word_contra_loss = self.loss(doc_word_sim_logits.transpose(2, 1), doc_word_labels)

        # doc_word_contra_loss = (doc_word_contra_loss * mask_src).sum() / doc_word_contra_loss.shape[1]
        # print(f"Doc word contrastive loss is {doc_word_contra_loss}")

        negative_graphs = [g for g in graph for _ in range(negative_num)]
        graph_node_features = graph_features.reshape(batch_size, nn+1, negative_num, -1)
        graph_node_features = graph_node_features.permute(0, 2, 1, 3)
        graph_node_features = graph_node_features.reshape(batch_size * negative_num * (nn+1), -1)
        # graph_node_idxes = torch.arange(0, batch_size * (nn+1) * negative_num, nn+1)
        graph_node_num = [num for num in node_num for _ in range(negative_num)]
        graph_node_idxes = np.cumsum(graph_node_num)
        graph_node_idxes = [torch.zeros_like(graph_node_idxes[0]).long().to(graph_node_features.device)] + graph_node_idxes.tolist()
        mask_indexes = []
        start = 0
        for num in graph_node_num:
            mask_indexes.extend([index + start for index in range(num)])
            start += nn + 1
        graph_node_features = graph_node_features[mask_indexes, :]
        graph_neighbor_features, graph_nodes_src = self.gnnEncoder(negative_graphs, graph_node_features, graph_node_idxes, graph_node_num)
        #print(graph_neighbor_features.shape)
        # all_neighbor_feat = []
        # all_nodes_src = []
        # for node_features in all_features:
        #     node_feature_res = []
        #     node_feature_idx = [0]
        #     for idx, node_feature in enumerate(node_features):
        #         n_num = node_num[idx]
        #         mask = torch.arange(n_num)
        #         node_feature_idx.append(node_feature_idx[-1] + len(mask))
                # node_feature_res.append(torch.index_select(node_feature, 0, torch.tensor(mask, device=node_feature.device)))
        #         node_feature_res.append(torch.index_select(node_feature, 0, mask.to(node_feature.device)))
        #    node_feature_res = torch.cat(node_feature_res, 0)
        #     assert len(node_feature_res) == sum(node_num).item()
            # graph = B x node_num * neighbur * negative + 1 * token_length
        #     print ("===========")
        #     neighbor_feat, nodes_src = self.gnnEncoder(graph, node_feature_res, node_feature_idx, node_num)
        #     all_neighbor_feat.append(neighbor_feat)
        #     all_nodes_src.append(nodes_src)

        overall_feat = graph_neighbor_features.reshape(negative_num, batch_size, nn+1, -1)
        # overall_feat = torch.stack(all_neighbor_feat) # (negative_num + 1) * batch_size * node_num * hidden_num
        overall_feat = overall_feat / (overall_feat.norm(dim=-1)[:, :, :, None]+1e-8)
        neighbor_feat = overall_feat[:, :, 1:, :]
        head_feat = overall_feat[:, :, 0:1, :]
        batch_size = neighbor_feat.size(1)
        nn = neighbor_feat.size(2)
        # negative_num = self.args.negative_number+1
        neighbor_feat = neighbor_feat.permute(1, 2, 0, 3).reshape(batch_size, nn * negative_num, -1)
        head_feat = head_feat.permute(1, 2, 0, 3).reshape(batch_size, negative_num, -1)
        cos_sim = torch.bmm(neighbor_feat, head_feat.permute(0, 2, 1))
        # batch_size * (node_num - 1 * (neightbor_num + 1)) * (1 * (neighbor_num + 1))

        # for i in range(self.args.negative_number+1):
        #     neighbor_feat = all_neighbor_feat[i] #batch_size*node_num*hidden_num

        #     pos_sim = self.cos(neighbor_feat[:,1:,:], neighbor_feat[:,0,:]).unsqueeze(2)
            #print(pos_sim.shape)
        #   sim = [pos_sim]
        #     all_id = list(range(self.args.negative_number+1))
        #     remain_negative = [each_i for each_i in all_id if each_i!=i]
        #     for j in remain_negative:
        #         each_sim = self.cos(neighbor_feat[:,1:,:], all_neighbor_feat[j][:,0,:]).unsqueeze(2)
        #         sim.append(each_sim)
        #     sim_logits = torch.cat(sim,2)/self.args.temp
        #     labels_graph = torch.zeros(sim_logits.shape[0],sim_logits.shape[1],device=neighbor_feat.device, dtype=torch.int64)
        #     each_contra_loss = self.loss(sim_logits.transpose(2,1), labels_graph)
        #     mask_node = (torch.arange(neighbor_node_num)[None, :]).to(device=neighbor_feat.device) < node_num[:, None]
                #print(mask_node.shape, sim_logits.shape, labels_graph.shape)
                #print(each_contra_loss)
        #     sim_logits_array.append((sim_logits.transpose(2, 1), labels_graph, mask_node))
        #     each_contra_loss = (each_contra_loss*mask_node.float()).sum()/each_contra_loss.shape[1]
        #     print (each_contra_loss)
        #     contra_loss += each_contra_loss
        # contra_loss = contra_loss/self.args.negative_number
        # print(f"Contrastive loss is {contra_loss}")
        #dec_state = self.decoder_with_graph.init_decoder_state(src, graph_nodes_src[0:1],)
        #decoder_outputs, state = self.decoder_with_graph(tgt[:, :-1], encoder_outputs, graph_neighbor_features[0:1], dec_state)
        dec_state = self.decoder.init_decoder_state(src, encoder_outputs)
        print (mask_src.shape)
        decoder_outputs, state = self.decoder(tgt[:, :-1], encoder_outputs, dec_state)
        return decoder_outputs, None, doc_word_cos_sim, cos_sim
