import argparse
import torch
import wandb
import torch.optim as optim
from models import Linear, MLP
from tqdm import tqdm
from utils.loss import partial_loss
from utils.eval import accuracy
from torch.utils.data import DataLoader
from datasets import TensorDataset
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from utils.data import partialize, mean_targets, instance_partialize
from sklearn.model_selection import train_test_split
import scipy
from utils.confpred import compute_quantile, generate_sets, efficiency, coverage

def main(args):
    print("data preparation")
    run = wandb.init(project="cp-bird", config=args, mode="online")

    if args.ds == "mnist":
        device = 'cpu'
        normalize = torchvision.transforms.Normalize((0.1307,), (0.3081,))
        
        train = torchvision.datasets.MNIST("./data/", train=True, download=True)
        train_x = torch.flatten(normalize(train.data.float()), start_dim=1)
        train_y = train.targets
        if args.noise == "instance":
            train_y_amb = F.one_hot(train.targets).float()
        else:
            train_y_amb = partialize(F.one_hot(train.targets).float(), args.p, args.q)
        train_x, cal_x, train_y_amb, cal_y_amb, train_y, cal_y = train_test_split(train_x, train_y_amb, train_y, test_size=args.n)
        train_dataset = TensorDataset(train_x,train_y_amb, train_y)     
        cal_dataset = TensorDataset(cal_x,cal_y_amb, cal_y)
        train_loader = DataLoader(train_dataset, batch_size=256)
        cal_loader = DataLoader(cal_dataset, batch_size=256)
        if args.noise == "instance":
            instance_partialize(train_loader, "mnist")
            instance_partialize(cal_loader, "mnist")
        test = torchvision.datasets.MNIST("./data/", train=False, download=True)
        test_x = torch.flatten(normalize(test.data.float()), start_dim=1)
        test_y = test.targets
        test_y_amb = partialize(F.one_hot(test.targets).float() , args.p, args.q)
        test_dataset = TensorDataset(test_x,test_y_amb, test_y)
        test_loader = DataLoader(test_dataset, batch_size=256)

        lr = 1e-2
        wd = 1e-3
        model = MLP()
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)

    elif args.ds == "kmnist":
        device = 'cpu'
        normalize = torchvision.transforms.Normalize((0.5,), (0.5,))
        
        train = torchvision.datasets.KMNIST("./data/", train=True, download=True)
        train_x = torch.flatten(normalize(train.data.float()), start_dim=1)
        train_y = train.targets
        if args.noise == "instance":
            train_y_amb = F.one_hot(train.targets).float()
        else:
            train_y_amb = partialize(F.one_hot(train.targets).float(), args.p, args.q)
        train_x, cal_x, train_y_amb, cal_y_amb, train_y, cal_y = train_test_split(train_x, train_y_amb, train_y, test_size=args.n)
        train_dataset = TensorDataset(train_x,train_y_amb, train_y)
        cal_dataset = TensorDataset(cal_x,cal_y_amb, cal_y)
        train_loader = DataLoader(train_dataset, batch_size=256)
        cal_loader = DataLoader(cal_dataset, batch_size=256)
        if args.noise == "instance":
            instance_partialize(train_loader, "kmnist")
            instance_partialize(cal_loader, "kmnist")
        
        test = torchvision.datasets.KMNIST("./data/", train=False, download=True)
        test_x = torch.flatten(normalize(test.data.float()), start_dim=1)
        test_y = test.targets
        test_y_amb = partialize(F.one_hot(test.targets).float() , args.p, args.q)
        test_dataset = TensorDataset(test_x,test_y_amb, test_y)
        test_loader = DataLoader(test_dataset, batch_size=256)


        lr = 1e-2
        wd = 1e-4
        model = MLP()
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)

    elif args.ds == "fmnist":
        device = 'cpu'
        normalize = torchvision.transforms.Normalize((0.1307,), (0.3081,))
        
        train = torchvision.datasets.FashionMNIST("./data/", train=True, download=True)
        train_x = torch.flatten(normalize(train.data.float()), start_dim=1)
        train_y = train.targets
        if args.noise == "instance":
            train_y_amb = F.one_hot(train.targets).float()
        else:
            train_y_amb = partialize(F.one_hot(train.targets).float(), args.p, args.q)
        train_x, cal_x, train_y_amb, cal_y_amb, train_y, cal_y = train_test_split(train_x, train_y_amb, train_y, test_size=args.n)
        train_dataset = TensorDataset(train_x,train_y_amb, train_y)
        cal_dataset = TensorDataset(cal_x,cal_y_amb, cal_y)
        train_loader = DataLoader(train_dataset, batch_size=256)
        cal_loader = DataLoader(cal_dataset, batch_size=256)
        if args.noise == "instance":
            instance_partialize(train_loader, "fmnist")
            instance_partialize(cal_loader, "fmnist")
        
        test = torchvision.datasets.FashionMNIST("./data/", train=False, download=True)
        test_x = torch.flatten(normalize(test.data.float()), start_dim=1)
        test_y = test.targets
        test_y_amb = partialize(F.one_hot(test.targets).float() , args.p, args.q)
        test_dataset = TensorDataset(test_x,test_y_amb, test_y)
        test_loader = DataLoader(test_dataset, batch_size=256)

        lr = 1e-2
        wd = 1e-5
        model = MLP()
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)

    elif args.ds == "lost":
        device = 'cpu'
        
        mat = scipy.io.loadmat('./data/lost.mat')
        mat_x = torch.from_numpy(mat["features"])
        mat_y = torch.argmax(torch.from_numpy(mat["logitlabels"]), dim=1)
        mat_y_amb = torch.from_numpy(mat["p_labels"]).float()
        mat_y_amb /= torch.sum(mat_y_amb, dim=1).repeat(mat_y_amb.shape[1], 1).T  
        train_x, test_x, train_y_amb, test_y_amb, train_y, test_y = train_test_split(mat_x, mat_y_amb, mat_y, test_size=0.2)
        train_x = F.normalize(train_x)
        test_x = F.normalize(test_x)
        train_x, cal_x, train_y_amb, cal_y_amb, train_y, cal_y = train_test_split(train_x, train_y_amb, train_y, test_size=args.n)
        
        train_dataset = TensorDataset(train_x,train_y_amb, train_y)
        cal_dataset = TensorDataset(cal_x,cal_y_amb, cal_y)
        train_loader = DataLoader(train_dataset, batch_size=100)
        cal_loader = DataLoader(cal_dataset, batch_size=100)
        test_dataset = TensorDataset(test_x,test_y_amb, test_y)
        test_loader = DataLoader(test_dataset, batch_size=100)

        n_in = train_x.shape[1]
        n_out = train_y_amb.shape[1]

        lr = 1e-1
        wd = 1e-10
        model = Linear(n_in, n_out)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

    elif args.ds == "msr":
        device = 'cpu'
        
        mat = scipy.io.loadmat('./data/MSRCv2.mat')
        mat_x = torch.from_numpy(mat["features"])
        mat_y = torch.argmax(torch.from_numpy(mat["logitlabels"]), dim=1)
        mat_y_amb = torch.from_numpy(mat["p_labels"]).float()
        mat_y_amb /= torch.sum(mat_y_amb, dim=1).repeat(mat_y_amb.shape[1], 1).T  
        train_x, test_x, train_y_amb, test_y_amb, train_y, test_y = train_test_split(mat_x, mat_y_amb, mat_y, test_size=0.2)
        train_x = F.normalize(train_x)
        test_x = F.normalize(test_x)
        train_x, cal_x, train_y_amb, cal_y_amb, train_y, cal_y = train_test_split(train_x, train_y_amb, train_y, test_size=args.n)
        
        train_dataset = TensorDataset(train_x,train_y_amb, train_y)
        cal_dataset = TensorDataset(cal_x,cal_y_amb, cal_y)
        train_loader = DataLoader(train_dataset, batch_size=100)
        cal_loader = DataLoader(cal_dataset, batch_size=100)
        test_dataset = TensorDataset(test_x,test_y_amb, test_y)
        test_loader = DataLoader(test_dataset, batch_size=100)

        n_in = train_x.shape[1]
        n_out = train_y_amb.shape[1]
        
        lr = 1e-2
        wd = 1e-6
        model = Linear(n_in, n_out)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    elif args.ds == "bird":
        device = 'cpu'
        
        mat = scipy.io.loadmat('./data/birdac.mat')
        mat_x = torch.from_numpy(mat["features"])
        mat_y = torch.argmax(torch.from_numpy(mat["logitlabels"]), dim=1)
        mat_y_amb = torch.from_numpy(mat["p_labels"]).float()
        mat_y_amb /= torch.sum(mat_y_amb, dim=1).repeat(mat_y_amb.shape[1], 1).T  
        train_x, test_x, train_y_amb, test_y_amb, train_y, test_y = train_test_split(mat_x, mat_y_amb, mat_y, test_size=0.2)
        mu = torch.mean(train_x, dim=0)
        sigma = torch.std(train_x, dim=0)
        train_x = (train_x - mu) / sigma 
        test_x = (test_x - mu) / sigma 
        # train_x = F.normalize(train_x)
        # test_x = F.normalize(test_x)
        train_x, cal_x, train_y_amb, cal_y_amb, train_y, cal_y = train_test_split(train_x, train_y_amb, train_y, test_size=args.n)
        
        
        train_dataset = TensorDataset(train_x,train_y_amb, train_y)
        cal_dataset = TensorDataset(cal_x,cal_y_amb, cal_y)
        train_loader = DataLoader(train_dataset, batch_size=100)
        cal_loader = DataLoader(cal_dataset, batch_size=100)
        test_dataset = TensorDataset(test_x,test_y_amb, test_y)
        test_loader = DataLoader(test_dataset, batch_size=100)

        n_in = train_x.shape[1]
        n_out = train_y_amb.shape[1]
        

        lr = 1e-2
        wd = 1e-10
        model = Linear(n_in, n_out)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    elif args.ds == "soccer":
        device = 'cpu'
        
        mat = scipy.io.loadmat('./data/spd.mat')
        mat_x = torch.from_numpy(mat["features"])
        mat_y = torch.argmax(torch.from_numpy(mat["logitlabels"]), dim=1)
        mat_y_amb = torch.from_numpy(mat["p_labels"]).float()
        mat_y_amb /= torch.sum(mat_y_amb, dim=1).repeat(mat_y_amb.shape[1], 1).T  
        train_x, test_x, train_y_amb, test_y_amb, train_y, test_y = train_test_split(mat_x, mat_y_amb, mat_y, test_size=0.2)
        mu = torch.mean(train_x, dim=0)
        sigma = torch.std(train_x, dim=0)
        train_x = (train_x - mu) / sigma 
        test_x = (test_x - mu) / sigma 
        # train_x = F.normalize(train_x)
        # test_x = F.normalize(test_x)
        train_x, cal_x, train_y_amb, cal_y_amb, train_y, cal_y = train_test_split(train_x, train_y_amb, train_y, test_size=args.n)
        
        train_dataset = TensorDataset(train_x,train_y_amb, train_y)
        cal_dataset = TensorDataset(cal_x,cal_y_amb, cal_y)
        train_loader = DataLoader(train_dataset, batch_size=100)
        cal_loader = DataLoader(cal_dataset, batch_size=100)
        test_dataset = TensorDataset(test_x,test_y_amb, test_y)
        test_loader = DataLoader(test_dataset, batch_size=100)

        n_in = train_x.shape[1]
        n_out = train_y_amb.shape[1]
        
        lr = 1e-2
        wd = 1e-10
        model = Linear(n_in, n_out)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    elif args.ds == "yahoo":
        device = 'cpu'
        
        mat = scipy.io.loadmat('./data/LYN.mat')
        mat_x = torch.from_numpy(mat["features"])
        mat_y = torch.argmax(torch.from_numpy(mat["logitlabels"]), dim=1)
        mat_y_amb = torch.from_numpy(mat["p_labels"]).float()
        mat_y_amb /= torch.sum(mat_y_amb, dim=1).repeat(mat_y_amb.shape[1], 1).T  
        train_x, test_x, train_y_amb, test_y_amb, train_y, test_y = train_test_split(mat_x, mat_y_amb, mat_y, test_size=0.2)
        train_x = F.normalize(train_x)
        test_x = F.normalize(test_x)
        train_x, cal_x, train_y_amb, cal_y_amb, train_y, cal_y = train_test_split(train_x, train_y_amb, train_y, test_size=args.n)
        
        train_dataset = TensorDataset(train_x,train_y_amb, train_y)
        cal_dataset = TensorDataset(cal_x,cal_y_amb, cal_y)
        train_loader = DataLoader(train_dataset, batch_size=100)
        cal_loader = DataLoader(cal_dataset, batch_size=100)
        test_dataset = TensorDataset(test_x,test_y_amb, test_y)
        test_loader = DataLoader(test_dataset, batch_size=100)

        n_in = train_x.shape[1]
        n_out = train_y_amb.shape[1]
        
        lr = 1e-2
        wd = 1e-10
        model = Linear(n_in, n_out)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


    wandb.log({"num_labels_all" : mean_targets(train_y_amb, "all")})
    wandb.log({"num_labels_amb" : mean_targets(train_y_amb, "amb")})

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.e)
    model = model.to(device)        
    print("training")
    for epoch in tqdm(range(args.e)):
        model.train()
        for inputs, targets, true, indexes in train_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.float())
            loss, new_targets = partial_loss(outputs, targets)   
            loss.backward()
            optimizer.step()
            for j, k in enumerate(indexes):
                train_loader.dataset.tensors[1][k,:] = new_targets[j,:].detach().float()
        
        scheduler.step()
        model.eval()
        train_acc = accuracy(model, train_loader)
        print(train_acc)
        test_acc = accuracy(model, test_loader)
        print(test_acc)
        wandb.log({"train_accuracy" : train_acc})
   
    test_acc = accuracy(model, test_loader)
    print(test_acc)

    wandb.log({"test_accuracy" : accuracy(model, test_loader)})

    print("conformal prediction")
    print("calibration")

    true_scores = []
    min_scores = []
    full_scores = []
    max_scores = []
    mean_scores = []
    mu_scores1 = []
    mu_scores2 = []
    mu_scores3 = []
    device = 'cpu'
    model.to(device)
    for inputs, targets, true_targets, _ in cal_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = F.softmax(model(inputs.float()), dim=1)
        for i in range(outputs.shape[0]):
            true_scores.append(1-outputs[i, true_targets[i]])
            candidates = outputs[i,targets[i,:]!=0]
            min_scores.append(1-torch.min(candidates))
            max_scores.append(1-torch.max(candidates))
            mean_scores.append(1-torch.mean(candidates))
            mu_scores1.append(1-(0.3 * torch.max(candidates) + (1-0.3) * torch.min(candidates)))
            mu_scores2.append(1-(0.5 * torch.max(candidates) + (1-0.5) * torch.min(candidates)))
            mu_scores3.append(1-(0.7 * torch.max(candidates) + (1-0.7) * torch.min(candidates)))
            for j in range(candidates.shape[0]):
                full_scores.append(1-candidates[j])

    true_q = compute_quantile(torch.tensor(true_scores), args.a)
    min_q = compute_quantile(torch.tensor(min_scores), args.a)
    full_q = compute_quantile(torch.tensor(full_scores), args.a)
    max_q = compute_quantile(torch.tensor(max_scores), args.a)
    mean_q = compute_quantile(torch.tensor(mean_scores), args.a)
    mu_q1 = compute_quantile(torch.tensor(mu_scores1), args.a)
    mu_q2 = compute_quantile(torch.tensor(mu_scores2), args.a)
    mu_q3 = compute_quantile(torch.tensor(mu_scores3), args.a)

    wandb.log({"true_q" : true_q})
    wandb.log({"min_q" : min_q})
    wandb.log({"full_q" : full_q})
    wandb.log({"max_q" : max_q})
    wandb.log({"mean_q" : mean_q})
    wandb.log({"mu_q" : [mu_q1, mu_q2, mu_q3]})

    print("testing")
    true_sets = []
    min_sets = []
    full_sets = []
    max_sets = []
    mean_sets = []
    mu_sets1 = []
    mu_sets2 = []
    mu_sets3 = []
    for inputs, _, targets, _ in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = F.softmax(model(inputs.float()), dim=1)
        true_sets.append(generate_sets(outputs, true_q))
        min_sets.append(generate_sets(outputs, min_q))
        full_sets.append(generate_sets(outputs, full_q))
        max_sets.append(generate_sets(outputs, max_q))
        mean_sets.append(generate_sets(outputs, mean_q))
        mu_sets1.append(generate_sets(outputs, mu_q1))
        mu_sets2.append(generate_sets(outputs, mu_q2))
        mu_sets3.append(generate_sets(outputs, mu_q3))

    true_sets = torch.concat(true_sets, dim=0)
    min_sets = torch.concat(min_sets, dim=0)
    full_sets = torch.concat(full_sets, dim=0)
    max_sets = torch.concat(max_sets, dim=0)
    mean_sets = torch.concat(mean_sets, dim=0)
    mu_sets1 = torch.concat(mu_sets1, dim=0)
    mu_sets2 = torch.concat(mu_sets2, dim=0)
    mu_sets3 = torch.concat(mu_sets3, dim=0)

    true_eff = efficiency(true_sets)
    true_cvg = coverage(true_sets, test_y)

    min_eff = efficiency(min_sets)
    min_cvg = coverage(min_sets, test_y)

    full_eff = efficiency(full_sets)
    full_cvg = coverage(full_sets, test_y)

    max_eff = efficiency(max_sets)
    max_cvg = coverage(max_sets, test_y)

    mean_eff = efficiency(mean_sets)
    mean_cvg = coverage(mean_sets, test_y)

    mu_eff1 = efficiency(mu_sets1)
    mu_cvg1 = coverage(mu_sets1, test_y)
    mu_eff2 = efficiency(mu_sets2)
    mu_cvg2 = coverage(mu_sets2, test_y)
    mu_eff3 = efficiency(mu_sets3)
    mu_cvg3 = coverage(mu_sets3, test_y)

    wandb.log({"true_eff" : true_eff})
    wandb.log({"true_cvg" : true_cvg})

    wandb.log({"min_eff" : min_eff})
    wandb.log({"min_cvg" : min_cvg})

    wandb.log({"full_eff" : full_eff})
    wandb.log({"full_cvg" : full_cvg})

    wandb.log({"max_eff" : max_eff})
    wandb.log({"max_cvg" : max_cvg})

    wandb.log({"mean_eff" : mean_eff})
    wandb.log({"mean_cvg" : mean_cvg})

    wandb.log({"mu_eff" : [mu_eff1, mu_eff2, mu_eff3]})
    wandb.log({"mu_cvg" : [mu_cvg1, mu_cvg2, mu_cvg3]})

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", type=int, default=500, help="Number of training epochs")
    parser.add_argument("-ds", type=str, default="mnist", help="Dataset")
    parser.add_argument("-p", type=float, default=1.0, help="Portion of data contaminated")
    parser.add_argument("-q", type=float, default=0.1, help="Probability of including label in candidate set")
    parser.add_argument("-n", type=float, default=0.1, help="Size of calibration set")
    parser.add_argument("-a", type=float, default=0.1, help="Error rate alpha of conformal prediction")
    parser.add_argument("-noise", type=str, default="random", help="Type of label noise: instance or random")
    
    args = parser.parse_args()
    main(args)