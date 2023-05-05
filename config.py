
def parser_add_main_args(parser):
    # Data
    parser.add_argument('--dataname', type=str, default='gamer')
    parser.add_argument('--num_masks', type=int, default=5, help='number of masks')
    parser.add_argument('--train_prop', type=float, default=.6, help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.2, help='validation label proportion')
    parser.add_argument('--test_prop', type=float, default=.2, help='test label proportion')

    # Model
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--L', type=int, default=2, help='number of conv layers')
    parser.add_argument('--K', type=int, default=5, help='Polynomial power')
    parser.add_argument('--dropout', type=float, default=0., help='dropout for MLP')
    parser.add_argument('--dprate', type=float, default=0., help='dropout for propagation layer')
    parser.add_argument('--gamma', type=float, default=.0, help='coef for regularizer')
    parser.add_argument('--gamma2', type=float, default=.0, help='coef for regularizer')
    parser.add_argument('--gamma3', type=float, default=.0, help='coef for regularizer')

    # Training
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--temp_lr', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--weight_decay', type=float, default=0.000)
    parser.add_argument('--seed', type=int, default=33780)

    # Case Study
    parser.add_argument('--mode', type=str, default='3', choices=['2', '3'], help='2: 2splits, 3: 3splits')
    parser.add_argument('--num_nodes', type=int, default=2000)
    parser.add_argument('--num_features', type=int, default=1000)
    parser.add_argument('--ratio', type=str, default='0.00', help='homophily ratio')
    parser.add_argument('--low', type=float, default=0.0)
    parser.add_argument('--middle', type=float, default=0.0)
    parser.add_argument('--high', type=float, default=0.0)

    # Baselines
    parser.add_argument('--gat_heads', type=int, default=1, help='number of heads for gat')
    parser.add_argument('--gpr_alpha', type=float, default=.1, help='alpha for gprgnn and appnp')
    parser.add_argument('--net', type=str, default='gprgnn')
    parser.add_argument('--hops', type=int, default=1, help='number of hops for mixhop')



