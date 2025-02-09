
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # experiment
    parser.add_argument("--exp_name", type=str, default="default",
                        help="exp name")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint path")
    
    # Parameters
    parser.add_argument("--alpha", type=float, default=1, 
                        help="alpha parameter of multi similarity")
    parser.add_argument("--beta", type=float, default=50, 
                        help="beta parameter of multi similarity")
    parser.add_argument("--base", type=float, default=0.0, 
                        help="base parameter of multi similarity")
    parser.add_argument("--eps", type=float, default=0.1, 
                        help="eps parameter of multi similarity miner")
    parser.add_argument("--miner_marg", type=float, default=0.2, 
                        help="margin parameter of triplet margin miner")
    parser.add_argument("--lr_adam", type=float, default=0.0001,
                        help="learning rate adam optimizer")
    parser.add_argument("--margin", type=float, default=0.1, 
                        help="Margin parameter of TripletMarginLoss")
    parser.add_argument("--swap", type=bool, default=False, 
                        help="swap parameter of TripletMarginLoss")
    parser.add_argument("--smooth", type=bool, default=False, 
                        help="smooth_loss parameter of TripletMarginLoss")
    parser.add_argument("--wd_adamw", type=float, default=0.01,
                        help="weight decay adam optimizer")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64,
                        help="The number of places to use per iteration (one place is N images)")
    parser.add_argument("--img_per_place", type=int, default=4,
                        help="The effective batch size is (batch_size * img_per_place)")
    parser.add_argument("--min_img_per_place", type=int, default=4,
                        help="places with less than min_img_per_place are removed")
    parser.add_argument("--max_epochs", type=int, default=1,
                        help="stop when training reaches max_epochs")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of processes to use for data loading / preprocessing")

    # Architecture parameters
    parser.add_argument("--descriptors_dim", type=int, default=512,
                        help="dimensionality of the output descriptors")
    parser.add_argument("--opt", type=str, default="sgd", 
                        help="type of optimizer")
    parser.add_argument("--loss", type=str, default="cl", 
                        help="type of loss function")
    parser.add_argument("--pool", type=str, default=None,
                        help="type of pooling")
    parser.add_argument("--miner", type=str, default=None,
                        help="type of miner")
    
    
    # Visualizations parameters
    parser.add_argument("--num_preds_to_save", type=int, default=0,
                        help="At the end of training, save N preds for each query. "
                        "Try with a small number like 3")
    parser.add_argument("--save_only_wrong_preds", action="store_true",
                        help="When saving preds (if num_preds_to_save != 0) save only "
                        "preds for difficult queries, i.e. with uncorrect first prediction")

    # Paths parameters
    parser.add_argument("--train_path", type=str, default="data/gsv_xs/train",
                        help="path to train set")
    parser.add_argument("--val_path", type=str, default="data/sf_xs/val",
                        help="path to val set (must contain database and queries)")
    parser.add_argument("--test_path", type=str, default="data/sf_xs/test",
                        help="path to test set (must contain database and queries)")
    
    args = parser.parse_args()
    return args

