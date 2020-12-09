import tensorflow as tf
import tensorflow_datasets as tfds

import torch
import torchvision
import torch.optim as optim

import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader

batch_size = int(1e2)

try:
    device = torch.device('cuda:0')
except:
    device = torch.device('cpu')
    
def reset_weights(model):
    
    #Function: reset_weights
    #Input:    model (Pytorch Model)
    #Process:  resets model weights
    #Example:  model.apply(DAL.reset_weights)
    #Output:   none
    
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear) or isinstance(model, nn.BatchNorm2d):
        model.reset_parameters()
        
def create_data_loader(images,labels):
    
    #Function: create_data_loader
    #Input:    images (Pytorch Tensor), 
    #          labels (Pytorch Tensor)
    #Process:  Creates dataloader
    #Example:  data_loader = create_data_loader(images,labels)
    #Output:   data_loader (Pytorch dataloader)
    
    dataset     = TensorDataset(images, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    
    return data_loader

def train_step(net,train_loader,test_loader,criterion,optimizer):
        
    #Function: train_step
    #Input:    net (Pytorch Model), 
    #          train_loader (Pytorch dataloader)
    #          test_loader (Pytorch dataloader)
    #          criterion (Pytorch loss function)
    #          optimizer (Pytorch optimizer)
    #Process:  training step
    #Example:  loss = train_step(net,train_loader,test_loader,criterion,optimizer)
    #Output:   np.mean(loss_array) (numpy value)
    
    loss_array     = []
    accuracy_array = []
    
    net.train()
    
    for i,data in enumerate(train_loader):
    
        image,label = data
        
        optimizer.zero_grad()

        out  = net(image.to(device))
        loss = criterion(out,label.long().to(device))
        loss.backward()

        optimizer.step()
        
        loss_array.append(loss.item())
    
    return np.mean(loss_array)
    
def train(net,train_loader,test_loader,criterion,optimizer):

    #Function: train
    #Input:    net (Pytorch Model), 
    #          train_loader (Pytorch dataloader)
    #          test_loader (Pytorch dataloader)
    #          criterion (Pytorch loss function)
    #          optimizer (Pytorch optimizer)
    #Process:  training process
    #Example:  loss_array = train(net,train_loader,test_loader,criterion,optimizer)
    #Output:   loss_array (list)
    
    numIters       = 25
    loss_array     = []

    for epoch in range(numIters):

        loss = train_step(net,train_loader,test_loader,criterion,optimizer)
        loss_array.append(loss)
                
    return loss_array

def check_accuracy(net,test_loader):
    
    #Function: check_accuracy
    #Input:    net (Pytorch Model), 
    #          test_loader (Pytorch dataloader)
    #Process:  check network accuracy
    #Example:  loss_array = check_accuracy(net,test_loader)
    #Output:   np.mean(accuracy_array) (numpy value)
    
    accuracy_array = []
    
    net.eval()
    
    for i,data in enumerate(test_loader):
        
        image,label = data
        
        out      = net(image.to(device))
        accuracy = (out.max(1)[1].cpu()==label).sum()/float(len(label))*100.0
        
        accuracy_array.append(accuracy.item())
    
    return np.mean(accuracy_array)

class Node():
    
    #Class:    Node
    #Input:    net (Pytorch Model), 
    #          data_points (numpy array)
    #          images (Pytorch Tensor)
    #          labels (Pytorch Tensor)
    #Value:    node within network
    
    def __init__(self,data_points,images,labels):
        
        self.data_points = data_points
        
        self.images = images
        self.labels = labels
        
        self.used_points = []
        
    def add_used_point(self,point):
            
        for ii in range(len(point)):
            if point[ii] in self.data_points and point[ii] not in self.used_points:
                self.used_points.append(point[ii])
        
    def reset_used_point(self):
        
        self.used_points = []
        
class Node_Network():
    
    #Class:    Node_Network
    #Input:    num_nodes (integer)
    #          train_images (Pytorch Tensor)
    #          train_labels (Pytorch Tensor)
    #          dim_out (integer)
    #Value:    network with nodes and hub
    
    def __init__(self,num_nodes,train_images,train_labels,dim_out):
        
        self.num_nodes    = num_nodes-1
        self.num_examples = int(5e4)//num_nodes
        self.distribution = np.random.randint(int(5e4),size = (self.num_examples,num_nodes))
        
        self.node    = []
        self.dim_out = dim_out
        
        self.alpha_hub  = np.zeros([self.num_examples,self.dim_out])
        self.alpha_node = np.zeros([self.num_nodes,self.num_examples,self.dim_out])

        for n in range(self.num_nodes):
            
            r = self.distribution[:,n]
            
            images = train_images[r,:,:,:]
            labels = train_labels[r]
            
            self.node.append(Node(r,images,labels))
        
        r = self.distribution[:,n+1]
        
        images = train_images[r,:,:,:]
        labels = train_labels[r]

        self.hub = Node(r,images,labels)
        
        self.images = train_images
        self.labels = train_labels
        
    def compression(self,net):
        
        self.alpha_values = torch.zeros([int(5e4),self.dim_out])
        
        for node_ii in range(self.num_nodes):    
            data_points                      = torch.Tensor(self.node[node_ii].data_points).long()
            image                            = self.images[data_points,:,:,:]
            
            self.alpha_values[data_points,:] = net(image.to(device)).cpu().detach()

        data_points                      = torch.Tensor(self.hub.data_points).long()
        image                            = self.images[data_points,:,:,:]

        self.alpha_values[data_points,:] = net(image.to(device)).cpu().detach() 
            
    def reset_data_points(self):
        
        for n in range(self.num_nodes):
            
            self.node[n].reset_used_point()
            
class Data_Optimizer():
    
    #Class:    Data_Optimizer
    #Input:    network (Node_Network object)
    #          iterations (integer)
    #Value:    optimizer for data selection from nodes
    
    def __init__(self,network,iterations):
        
        self.network = network

        self.w   = torch.randn([int(5e4),1],requires_grad=False)
        
        self.x   = torch.randn([int(5e4),1],requires_grad=True)
        self.y   = torch.randn([int(5e4),1],requires_grad=False)
        self.s   = torch.randn([int(5e4),1],requires_grad=False)

        self.theta = 1
        self.lambda_val = 10
        
        self.iterations = iterations
        self.optimizer  = optim.SGD([self.x],lr=1e-2,weight_decay=0)
        
        self.removed_points = torch.Tensor([]).long()
        
    def update_values(self,network):
        
        self.A = torch.matmul((self.network.alpha_values - self.network.alpha_values.mean(0)),((self.network.alpha_values - self.network.alpha_values.mean(0))).transpose(0,1))
        self.P = (self.theta*torch.eye(int(5e4)) + self.A)#.to(device)
        
    def Lagrangian(self):
        
        f = - torch.norm(torch.matmul(torch.sigmoid(self.x).transpose(0,1),(self.network.alpha_values - self.network.alpha_values.mean(0))))
        g = self.lambda_val*torch.norm(self.y,1)
        
        C = (self.theta/2)*torch.norm(self.x-self.y+self.s,2) - (self.theta/2)*torch.norm(self.s,2)
        
        return f + g + C

    def optimize(self):
                        
        for k in range(self.iterations):
            
            Loss = self.Lagrangian()
            
            Loss.backward()
            self.optimizer.step()
            
            with torch.no_grad():

                self.y = F.softshrink((self.theta/self.lambda_val)*(self.x+self.s),(self.theta/self.lambda_val))
                self.s = self.s + (self.x - self.y)
                
                if torch.norm(self.x - self.y) < float(1e-8):
                    break
                    
            Loss = self.Lagrangian()
            
        self.w = self.x
        
    def select_points(self):
        
        chosen_points = torch.zeros([self.network.num_nodes]).long()
        
        W = self.w.squeeze()
        W = W.detach()
        
        if len(self.removed_points) > 0:
            W[self.removed_points] = - int(1e8)*torch.ones([len(self.removed_points)])
            
        for ii in range(self.network.num_nodes):
            chosen_points[ii] = (W[self.network.num_examples*(ii):self.network.num_examples*(ii+1)].argmax()+self.network.num_examples*(ii)).long()
        
        self.removed_points = torch.cat((self.removed_points,chosen_points),0)
        
        return chosen_points
    
def grab_examples(network,data_selector,hub_images,hub_labels,num_examples_increase):

    #Function: grab_examples_random
    #Input:    network (Node_Network Object), 
    #          hub_images (Pytorch Tensor)
    #          hub_labels (Pytorch Tensor)
    #Process:  gives chosen examples from each node based on Data_Optimizer
    #Example:  hub_images,hub_labels = grab_examples_random(network,hub_images,hub_labels,random_points)
    #Output:   hub_images (Pytorch Tensor)
    #          hub_labels (Pytorch Tensor)
    
    new_images = torch.zeros([network.num_nodes,3,32,32])
    new_labels = torch.zeros([network.num_nodes])
        
    data_selector.update_values(network)
    data_selector.optimize()
    
    new_alpha = torch.Tensor([])
    var_image = torch.Tensor([])
    
    for ii in range(num_examples_increase):
        
        selected_points = data_selector.select_points()

        new_images = network.images[selected_points,:,:,:]
        new_labels = network.labels[selected_points]
        
        hub_images,hub_labels = add_points_to_hub(hub_images,hub_labels,new_images,new_labels)

        for n in range(network.num_nodes):

            network.node[n].add_used_point(selected_points)
                
    return hub_images,hub_labels

def grab_examples_random(network,hub_images,hub_labels,random_points):

    #Function: grab_examples_random
    #Input:    network (Node_Network Object), 
    #          hub_images (Pytorch Tensor)
    #          hub_labels (Pytorch Tensor)
    #          random_points (numpy array)
    #Process:  gives random examples from each node
    #Example:  hub_images,hub_labels = grab_examples_random(network,hub_images,hub_labels,random_points)
    #Output:   hub_images (Pytorch Tensor)
    #          hub_labels (Pytorch Tensor)
    
    new_images = torch.zeros([network.num_nodes,3,32,32])
    new_labels = torch.zeros([network.num_nodes])
    
    new_images = network.images[random_points,:,:,:]
    new_labels = network.labels[random_points]
    
    hub_images,hub_labels = add_points_to_hub(hub_images,hub_labels,new_images,new_labels)
                
    return hub_images,hub_labels

def add_points_to_hub(current_images,current_labels,new_images,new_labels):
    
    #Function: add_points_to_hub
    #Input:    current_images (Pytorch Tensor)
    #          current_labels (Pytorch Tensor)
    #          new_images (Pytorch Tensor)
    #          new_labels (Pytorch Tensor)
    #Process:  concatenates two tensors
    #Example:  hub_images,hub_labels = add_points_to_hub(current_images,current_labels,new_images,new_labels)
    #Output:   hub_images (Pytorch Tensor)
    #          hub_labels (Pytorch Tensor)
    
    hub_images = torch.zeros([len(current_labels)+len(new_labels),3,32,32])
    hub_labels = torch.zeros([len(current_labels)+len(new_labels)])
    
    hub_images[:len(current_labels),:,:,:] = current_images
    hub_images[len(current_labels):,:,:,:] = new_images
    
    hub_labels[:len(current_labels)] = current_labels
    hub_labels[len(current_labels):] = new_labels
    
    return hub_images,hub_labels

def train_model(net,hub_images,hub_labels,test_loader,criterion,optimizer):
                
    #Function: train_model
    #Input:    net (Pytorch Model)
    #          hub_images (Pytorch Tensor)
    #          hub_labels (Pytorch Tensor)
    #          test_loader (Pytorch dataloader)
    #          criterion (Pytorch loss function)
    #          optimizer (Pytorch optimizer)
    #Process:  trains model
    #Example:  train_model(net,hub_images,hub_labels,test_loader,criterion,optimizer)
    #Output:   none
    
    torch.save(net.state_dict(), r'untrain_model.pt')
    
    hub_loader = create_data_loader(hub_images,hub_labels)
    loss       = train(net,hub_loader,test_loader,criterion,optimizer)
    
    torch.save(net.state_dict(), r'train_model.pt')
    
def run_test(net,data_selector,test_loader,criterion,optimizer,iterations,num_examples_increase,network,hub_images,hub_labels):

    #Function: run_test
    #Input:    net (Pytorch Model)
    #          data_selector (Data_Optimizer Object)
    #          test_loader (Pytorch dataloader)
    #          criterion (Pytorch loss function)
    #          optimizer (Pytorch optimizer)
    #          iterations (integer)
    #          num_examples_increase (integer)
    #          network (Node_Network object)
    #          hub_images (Pytorch Tensor)
    #          hub_labels (Pytorch Tensor)
    #Process:  runs test with Data_Optimizer Object
    #Example:  loss_array,accuracy_array = run_test(net,iterations,num_examples_increase,network,hub_images,hub_labels)
    #Output:   loss_array (numpy array)
    #          accuracy_array (numpy array)
    
    loss_array     = []
    accuracy_array = []
    alpha_array    = torch.Tensor([])
    var_img_array  = torch.Tensor([])

    for iter_num in range(iterations):

        hub_loader = create_data_loader(hub_images,hub_labels)

        net.load_state_dict(torch.load(r'untrain_model.pt'))
        network.compression(net)
        net.load_state_dict(torch.load(r'train_model.pt'))

        accuracy = check_accuracy(net,test_loader)
        accuracy_array.append(accuracy)

        hub_images,hub_labels = grab_examples(network,data_selector,hub_images,hub_labels,num_examples_increase)

        net.load_state_dict(torch.load(r'untrain_model.pt'))
        
    hub_loader    = create_data_loader(hub_images,hub_labels)
    alpha_array   = output_alpha(net,hub_loader,hub_labels)
    var_img_array = hub_images.reshape(-1,3*32*32).var(0)
    
    loss = train(net,hub_loader,test_loader,criterion,optimizer)
    network.compression(net)

    loss_array.append(loss[len(loss)-1])

    accuracy = check_accuracy(net,test_loader)
    accuracy_array.append(accuracy)
    
    print('---------------------------------Optimization Output Final---------------------------------')
    print()
    print()
    print('Loss: ' + str(round(loss[len(loss)-1],2)))
    print('Accuracy: ' + str(accuracy))
    print()
    print()

    return loss_array,accuracy_array,alpha_array,var_img_array

def sample_random_points(network,iterations,num_examples_increase):

    #Function: sample_random_points
    #Input:    network (Node_Network object)
    #          iterations (integer)
    #          num_examples_increase (integer)
    #Process:  creates random sampling schedule for each of the nodes
    #Example:  random_list = sample_random_points(network,iterations,num_examples_increase)
    #Output:   random_list (numpy array)
    
    random_list = np.zeros([network.num_nodes,num_examples_increase,iterations])

    for n in range(network.num_nodes):

        r = network.node[n].data_points
        np.random.shuffle(r)

        num = 0

        for ee in range(num_examples_increase):
            for ii in range(iterations):

                random_list[n,ee,ii] = r[num]

                num += 1
        
    return random_list

def output_alpha(net,hub_loader,hub_labels):
    
    #Function: sample_random_points
    #Input:    network (Node_Network object)
    #          iterations (integer)
    #          num_examples_increase (integer)
    #Process:  creates random sampling schedule for each of the nodes
    #Example:  random_list = sample_random_points(network,iterations,num_examples_increase)
    #Output:   random_list (numpy array)
    
    alpha_values = torch.zeros(len(hub_labels),10)
    
    net.eval()
    
    n = 0
    for i,data in enumerate(hub_loader):
        image,label = data
        out         = net(image.to(device)).detach().cpu()
        for ii in range(len(out)):
            alpha_values[n,:] = out[ii,:]
            n += 1
            
    return alpha_values
    
def run_test_random(net,test_loader,criterion,optimizer,iterations,num_examples_increase,network,hub_images,hub_labels,random_list):

    #Function: run_test_random
    #Input:    net (Pytorch Model)
    #          test_loader (Pytorch dataloader)
    #          test_loader (Pytorch dataloader)
    #          criterion (Pytorch loss function)
    #          optimizer (Pytorch optimizer)
    #          iterations (integer)
    #          num_examples_increase (integer)
    #          network (Node_Network object)
    #          hub_images (Pytorch Tensor)
    #          hub_labels (Pytorch Tensor)
    #          random_list (numpy array)
    #Process:  runs test with random sampling
    #Example:  loss_array,accuracy_array = run_test_random(net,test_loader,criterion,optimizer,iterations,num_examples_increase,network,hub_images,hub_labels,random_list)
    #Output:   loss_array (numpy array)
    #          accuracy_array (numpy array)
    
    loss_array     = []
    accuracy_array = []
    
    for iter_num in range(iterations):

        hub_loader = create_data_loader(hub_images,hub_labels)

        net.load_state_dict(torch.load(r'untrain_model.pt'))
        network.compression(net)
        net.load_state_dict(torch.load(r'train_model.pt'))
        
        accuracy = check_accuracy(net,test_loader)
        accuracy_array.append(accuracy)

        for i in range(num_examples_increase):
            r = random_list[:,i,iter_num]
            hub_images,hub_labels = grab_examples_random(network,hub_images,hub_labels,r)
        
        net.load_state_dict(torch.load(r'untrain_model.pt'))
        
    hub_loader    = create_data_loader(hub_images,hub_labels)
    alpha_array   = output_alpha(net,hub_loader,hub_labels)
    var_img_array = hub_images.reshape(-1,3*32*32).var(0)
    
    loss = train(net,hub_loader,test_loader,criterion,optimizer)
    network.compression(net)

    loss_array.append(loss[len(loss)-1])

    accuracy = check_accuracy(net,test_loader)
    accuracy_array.append(accuracy)

    print('---------------------------------Random Output Final---------------------------------')
    print()
    print()
    print('Loss: ' + str(round(loss[len(loss)-1],2)))
    print('Accuracy: ' + str(accuracy))
    print()
    print()

    return loss_array,accuracy_array,alpha_array,var_img_array
