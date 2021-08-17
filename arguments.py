import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='If training is to be done on a GPU')
    parser.add_argument('--dataset', type=str, default='cifar100', help='Name of the dataset used.')
    parser.add_argument('--classes', type=int, default=100, help='Number of classes in the target model.')
    parser.add_argument('--batch_size', type=int, default=250, help='Batch size used for training and testing')
    parser.add_argument('--train_iterations', type=int, default=20000, help='Number of training iterations') #20000
    parser.add_argument('--task_epochs', type=int, default=200, help='Number of training iterations') #200
    parser.add_argument('--latent_dim', type=int, default=200, help='The dimensionality of the VAE latent dimension')
    parser.add_argument('--data_path', type=str, default='../data', help='Path to where the data is')
    parser.add_argument('--beta', type=float, default=1, help='Hyperparameter for training. The parameter for VAE')
    parser.add_argument('--num_adv_steps', type=int, default=1, help='Number of adversary steps taken for every task model step')
    parser.add_argument('--num_vae_steps', type=int, default=2, help='Number of VAE steps taken for every task model step')
    parser.add_argument('--adversary_param', type=float, default=1, help='Hyperparameter for training. lambda2 in the paper')
    parser.add_argument('--log_name', type=str, default='accuracies.log', help='Final performance of the models will be saved with this name')
    parser.add_argument('--initnumber', type=int, default=5000, help='init samples to be labeled')
    parser.add_argument('--random_version', type=int, default=8, help='initial version to be used')
    parser.add_argument('--randompath', type=str, default='../random/', help='initial version to be used')
    parser.add_argument('--resultpath', type=str, default='./results_cifar100/', help='result for THE model')
    
    
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
    if not os.path.exists(args.resultpath):
        os.mkdir(args.resultpath)
    
    return args


