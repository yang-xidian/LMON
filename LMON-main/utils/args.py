from argparse import ArgumentParser
def make_args():
    parser = ArgumentParser()
    # general
    parser.add_argument('--comment', dest='comment', default='0', type=str,
                        help='comment')
    parser.add_argument('--task', dest='task', default='link', type=str,
                        help='link; link_pair')
    parser.add_argument('--model', dest='model', default='GraphReach', type=str,
                        help='model class name. E.g. GraphReach, GCN, SAGE, ...')
    parser.add_argument('--gpu', dest='gpu', action='store_true',
                        help='whether use gpu')
    parser.add_argument('--cache_no', dest='cache', action='store_false',
                        help='whether use cache')
    parser.add_argument('--cpu', dest='gpu', action='store_false',
                        help='whether use cpu')
    parser.add_argument('--cuda', dest='cuda', default='0', type=str)
    parser.add_argument('--attention', dest='attention', action='store_true',
                        help='whether use attention')

    # dataset
    parser.add_argument('--remove_link_ratio', dest='remove_link_ratio', default=0.2, type=float)
    parser.add_argument('--rm_feature', dest='rm_feature', action='store_true',
                        help='whether rm_feature')
    parser.add_argument('--rm_feature_no', dest='rm_feature', action='store_false',
                        help='whether rm_feature')
    parser.add_argument('--permute', dest='permute', action='store_true',
                        help='whether permute subsets')
    parser.add_argument('--permute_no', dest='permute', action='store_false',
                        help='whether permute subsets')
    parser.add_argument('--feature_pre', dest='feature_pre', action='store_true',
                        help='whether pre transform feature')
    parser.add_argument('--feature_pre_no', dest='feature_pre', action='store_false',
                        help='whether pre transform feature')
    parser.add_argument('--select_anchors', dest='select_anchors', default='DiversifiedRandomK', type=str,
                        help='DiversifiedRandomK;DiversifiedTopK;topK; random')

    parser.add_argument('--batch_size', dest='batch_size', default=8, type=int) # implemented via accumulating gradient
    parser.add_argument('--layer_num', dest='layer_num', default=2, type=int)
    parser.add_argument('--feature_dim', dest='feature_dim', default=32, type=int)
    parser.add_argument('--hidden_dim', dest='hidden_dim', default=32, type=int)
    parser.add_argument('--output_dim', dest='output_dim', default=32, type=int)
    parser.add_argument('--anchor_num', dest='anchor_num', default=64, type=int)
    parser.add_argument('--normalize_adj', dest='normalize_adj', action='store_true',
                        help='whether normalize_adj')

    parser.add_argument('--epoch_num', dest='epoch_num', default=2001, type=int)
    parser.add_argument('--repeat_num', dest='repeat_num', default=2, type=int) # 10
    parser.add_argument('--epoch_log', dest='epoch_log', default=10, type=int)

    parser.add_argument('--approximate', dest='approximate', default=-1, type=int,
                        help='k-hop shortest path distance. -1 means exact shortest path') # -1, 2
                        
    #NODE2VEC ARGUMENTS
    parser.add_argument('--fastRandomWalk', dest='fastRandomWalk', action='store_true',
                        help='Default is NormalRandomWalk.')
    parser.set_defaults(fastRandomWalk=False)

    parser.add_argument('--attentionAddSelf', dest='attentionAddSelf', action='store_true',
                        help='Default is False.')
    parser.set_defaults(attentionAddSelf=False)
    
    parser.add_argument('--walk_length', type=int, default=30,
                    help='Length of walk per source. Default is 80.')

    parser.add_argument('--num_walks', type=int, default=1, #50
                        help='Number of walks per source. Default is 10.')
    parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weightedRandomWalk', dest='weightedRandomWalk', action='store_true',
                        help='Default is unweightedRandomWalk.')
    parser.add_argument('--unweightedRandomWalk', dest='weightedRandomWalk', action='store_false')
    parser.set_defaults(weightedRandomWalk=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    parser.add_argument('--normalized', dest='normalized', action='store_true',
                        help='Boolean specifying (un)normalized. Default is unnormalized.')
    parser.add_argument('--unnormalized', dest='normalized', action='store_false')
    parser.set_defaults(normalized=False)

    parser.add_argument('--edgelabel', dest='edgelabel', action='store_true',
                        help='whether use edgelabel')
    parser.add_argument('--noedgelabel', dest='edgelabel', action='store_false',
                        help='whether use edgelabel')

    parser.add_argument('--sampleXwalks', type=float, default=0.3,
                        help='Number of walks to be sampled for topK anchors')
    parser.add_argument('--sampleMbigraphs', type=int, default=5,
                        help='Number of bigraphs to be sampled')

    parser.add_argument('--deleteFedges', type=float, default=0.01,
                        help='Fraction of edges deleted to be deleted for Adversarial Attack')
    parser.add_argument('--addFedges', type=float, default=0.10,
                        help='Add false edges to nodes involved in sampled fraction of test pairs')

    parser.add_argument('--AdversarialAttack', dest='AdversarialAttack', action='store_true',
                        help='Boolean flag')

    parser.add_argument('--Num_Anchors', dest='Num_Anchors', default='logn2', type=str,
                        help='Num_Anchors')

    parser.set_defaults(gpu=True, task='link', model='GraphReach', dataset='All',select_anchors='DiversifiedRandomK',
                        cache=False, rm_feature=False,
                        permute=False, feature_pre=True, dropout=True,
                        normalize_adj=False,edgelabel=False,attention=False,AdversarialAttack=False)
    
    # Training parameter
    parser.add_argument('--ptb_rate', type=float, default=0, help='if lploss')
    parser.add_argument('--bias_init', type=float, default=0, help='if lploss')
    parser.add_argument('--gamma', type=float, default=0, help='if lploss')
    parser.add_argument('--k', type=int, default=5, help='save embeddings')

    parser.add_argument('--train_size', type=int, default=0, help='if plot')
    parser.add_argument('--pca', type=int, default=0, help='if plot')

    parser.add_argument('--ssl', type=str, default=None, help='ssl agent')
    parser.add_argument('--lambda_', type=float, default=0, help='if lploss')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Disable validation during training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=800,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='Initial learning rate.')
    parser.add_argument('--lradjust', action='store_true',
                        default=False, help='Enable leraning rate adjust.(ReduceLROnPlateau or Linear Reduce)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--mixmode", action="store_true",
                        default=False, help="Enable CPU GPU mixing mode.")
    parser.add_argument("--warm_start", default="",
                        help="The model name to be loaded for warm start.")
    parser.add_argument('--debug', action='store_true',
                        default=False, help="Enable the detialed training output.")
    parser.add_argument('--dataset', default="cora", help="The data set")
    parser.add_argument('--datapath', default="data/", help="The data path.")
    parser.add_argument("--early_stopping", type=int,
                        default=0, help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")
    parser.add_argument("--no_tensorboard", default=True, help="Disable writing logs to tensorboard")

    # Model parameter
    parser.add_argument('--type',
                        help="Choose the model to be trained.(mutigcn, resgcn, densegcn, inceptiongcn)")
    parser.add_argument('--inputlayer', default='gcn',
                        help="The input layer of the model.")
    parser.add_argument('--outputlayer', default='gcn',
                        help="The output layer of the model.")
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--withbn', action='store_true', default=False,
                        help='Enable Bath Norm GCN')
    parser.add_argument('--withloop', action="store_true", default=False,
                        help="Enable loop layer GCN")
    parser.add_argument('--nhiddenlayer', type=int, default=1,
                        help='The number of hidden layers.')
    parser.add_argument("--normalization", default="AugNormAdj",
                        help="The normalization on the adj matrix.")
    parser.add_argument("--sampling_percent", type=float, default=1.0,
                        help="The percent of the preserve edges. If it equals 1, no sampling is done on adj matrix.")
    # parser.add_argument("--baseblock", default="res", help="The base building block (resgcn, densegcn, mutigcn, inceptiongcn).")
    parser.add_argument("--nbaseblocklayer", type=int, default=1,
                        help="The number of layers in each baseblock")
    parser.add_argument("--aggrmethod", default="default",
                        help="The aggrmethod for the layer aggreation. The options includes add and concat. Only valid in resgcn, densegcn and inecptiongcn")
    parser.add_argument("--task_type", default="full", help="The node classification task type (full and semi). Only valid for cora, citeseer and pubmed dataset.")


    
    args = parser.parse_args()
    return args