import scipy.io as sio                     
import numpy as np                         
import matplotlib.pyplot as plt
from yaml import DirectiveToken           
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, Tanh, BatchNorm1d as BN
import wireless_networks_generator as wg
import helper_functions
import time

class init_parameters():
    def __init__(self):
        # Wireless network settings
        self.n_links = train_K       
        self.field_length = 500
        self.shortest_directLink_length = 2
        self.longest_directLink_length = 50
        self.shortest_crossLink_length = 2
        self.bandwidth = 5e6
        self.carrier_f = 2.4e9
        self.tx_height = 2
        self.rx_height = 2
        self.antenna_gain_decibel = 9
        self.tx_power_milli_decibel = 40
        self.tx_power = np.power(10, (self.tx_power_milli_decibel-30)/10)
        self.noise_density_milli_decibel = -174
        self.input_noise_power = np.power(10, ((self.noise_density_milli_decibel-30)/10)) * self.bandwidth
        self.output_noise_power = self.input_noise_power
        self.setting_str = "{}_links_{}X{}_{}_{}_length".format(self.n_links, self.field_length, self.field_length, self.shortest_directLink_length, self.longest_directLink_length)

def get_directLink_csi(csi):
    directlink_csi = np.diagonal(csi[:,:,:,:,0], axis1=2, axis2=3)
    directlink_csi = np.expand_dims(directlink_csi,axis=3)
    for i in range(Nt-1):
        antenna = np.diagonal(csi[:,:,:,:,i+1], axis1=2, axis2=3)
        antenna = np.expand_dims(antenna,axis=3)
        directlink_csi = np.concatenate((directlink_csi, antenna), axis=3)
    return directlink_csi

def normalize_directlink_data(train_directlink_data,test_directlink_data):
    train_copy = np.copy(train_directlink_data)
    train_mean = np.sum(train_copy)/train_layouts/frame_num/train_K/Nt
    train_var = np.sqrt(np.sum(np.square(train_copy-train_mean))/train_layouts/frame_num/train_K/Nt)
    norm_train = (train_directlink_data - train_mean)/train_var
    norm_test = (test_directlink_data - train_mean)/train_var 
    return norm_train, norm_test

def normalize_agg_constants(train_data):
    mask = np.eye(train_K)
    norm_mean = 0
    norm_var = 0
    for i in range(Nt):
        train_copy = np.copy(train_data[:,:,:,:,i])    
        diag_H = np.multiply(mask,train_copy)
        off_diag = train_copy - diag_H
        off_diag = abs(off_diag)*abs(off_diag)
        off_diag_mean = np.sum(off_diag)/train_layouts/frame_num/train_K/(train_K-1)
        off_diag_var = np.sqrt(np.sum(np.square(off_diag-off_diag_mean))/train_layouts/frame_num/train_K/(train_K-1))
        norm_mean = norm_mean+off_diag_mean
        norm_var = norm_var +off_diag_var
    return norm_mean, norm_var

def proc_data(HH, norm_real,norm_imag, K):
    n = HH.shape[0]
    data_list = []
    for i in range(n):
            data = build_graph_sequence(HH[i,:,:,:,:], norm_real[i,:,:,:], norm_imag[i,:,:,:],K)
            data_list.append(data)
    return data_list

def build_graph_sequence(csis, norm_real, norm_imag, K):
    n = csis.shape[0]
    # For the last dimension of x, first 2Nt repsents directlink CSI for the previous frame
    # 2Nt:4Nt represents directlink CSI for the current frme
    # 4Nt: represents graph embdding
    # Note that the directlink CSI for the previous frame will be used for pilot beamformer,
    # the directlink CSI for the current frame will be used for data beamfomer and graph embedding update
    x = np.zeros((n,K,Nt*4+graph_embedding_size))
    adj = np.zeros((n,2,K*(K-1)))
    edge_attr = np.zeros((n,K*(K-1),Nt*2))
    y = np.zeros((n,1,K,K,Nt*2))
    for i in range(n):
        if i == 0:
            x1r = norm_real[i,:]
            x1i = norm_imag[i,:]
        else:
            x1r = norm_real[i-1,:]
            x1i = norm_imag[i-1,:]
        x2r = norm_real[i,:]
        x2i = norm_imag[i,:]
        x3 = np.zeros((K,graph_embedding_size))
        x[i,:,:] = np.concatenate((x1r,x1i,x2r,x2i,x3),axis=1)
    
        #Consider fully connected graph
        csi_copy = np.copy(csis[i,:,:,0])
        mask = np.eye(K)
        diag_csi_copy = np.multiply(mask,csi_copy)
        csi_copy = csi_copy - diag_csi_copy
        attr_ind = np.nonzero(csi_copy)

        csi_copy = np.copy(csis[i,:,:,:])
        edge_attr_real = np.real(csi_copy[attr_ind])
        edge_attr_imag = np.imag(csi_copy[attr_ind])

        edge_attr_tmp = np.concatenate((edge_attr_real,edge_attr_imag), axis=1)
        edge_attr[i:,:] = edge_attr_tmp
    
        attr_ind = np.array(attr_ind)
        adj[i,0,:] = attr_ind[1,:]
        adj[i,1,:] = attr_ind[0,:]

        H1 = np.real(csis[i,:,:,:])
        H2 = np.imag(csis[i,:,:,:])
        HH = np.concatenate((H1,H2),axis=-1)
        y[i,:,:,:] = np.expand_dims(HH,axis=0)

    x = torch.tensor(x, dtype=torch.float)
    x = torch.transpose(x,0,1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_attr = torch.transpose(edge_attr,0,1)
    edge_index = torch.tensor(adj, dtype=torch.long)
    edge_index = torch.transpose(edge_index,0,1)
    y = torch.tensor(y, dtype=torch.float)
    y = torch.transpose(y,0,1)
    data = Data(x=x, edge_index=edge_index.contiguous(),edge_attr = edge_attr, y = y)
    return data

class PG4UConv(MessagePassing):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(PG4UConv, self).__init__(aggr='add', **kwargs)
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        
    def update(self, aggr_out, x):
        aggr_out_norm = (aggr_out - norm_aggOTA_mean)/norm_aggOTA_var
        # Use directlink CSI in the current frme and graph embedding for update
        tmp = torch.cat([x[:,2*Nt:], aggr_out_norm], dim=1)
        comb = self.mlp2(tmp)
        return comb
        
    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
         # Use directlink CSI in the previous frme and graph embedding for generating pilot beamformer
        tmp = torch.cat([x_j[:,:2*Nt],x_j[:,4*Nt:]], dim=1)
        agg = self.mlp1(tmp)

        # Normalization of pilot beamformer due to power constraint
        nor = torch.sqrt(torch.sum(torch.mul(agg,agg),axis=1))
        nor = nor.unsqueeze(axis=-1)
        comp1 = torch.ones(agg.size(), device=device)
        agg = torch.div(agg,torch.max(comp1,nor) )

        # Compute received power from each link, used for aggregation
        rx_power1 = torch.mul(edge_attr[:,:Nt], agg[:,:Nt])       
        rx_power1 = torch.sum(rx_power1,axis=-1)
        rx_power2 = torch.mul(edge_attr[:,Nt:], agg[:,Nt:])
        rx_power2 = torch.sum(rx_power2,axis=-1)
        rx_power3 = torch.mul(edge_attr[:,:Nt], agg[:,Nt:])
        rx_power3 = torch.sum(rx_power3,axis=-1)
        rx_power4 = torch.mul(edge_attr[:,Nt:], agg[:,:Nt])
        rx_power4 = torch.sum(rx_power4,axis=-1)
        rx_power = torch.mul(rx_power1 + rx_power2,rx_power1 + rx_power2) + torch.mul(rx_power3 - rx_power4,rx_power3 - rx_power4)
        rx_power = rx_power.unsqueeze(-1)
        return rx_power

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1,self.mlp2)

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels))
    ])
class PG4U(torch.nn.Module):
    def __init__(self):
        super(PG4U, self).__init__()
        self.hidden_size = graph_embedding_size    
        self.h2o = MLP([graph_embedding_size, 32])
        self.h2o = Seq(*[self.h2o,Seq(Lin(32, 2*Nt, bias = True), Tanh())])
        self.tanh = nn.Tanh()
        self.mlp1 = MLP([graph_embedding_size+2*Nt, 32, 32])
        self.mlp1 = Seq(*[self.mlp1,Seq(Lin(32, 2*Nt, bias = True), Tanh())])
        self.mlp2 = MLP([1+2*Nt+graph_embedding_size, 32, graph_embedding_size])
        self.conv = PG4UConv(self.mlp1,self.mlp2)

    def forward(self, data):
        hidden = Variable(torch.zeros(links*batches, self.hidden_size))
        hidden = hidden.to(device)
        outputs = torch.zeros(frames-1,links*batches,2*Nt)
        outputs = outputs.to(device)
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        # Ignore the first frame in PG4U since beamformers depend on the CSI in the previous frame
        for t in range(frames-1):
            x_input = x[:,t,:]
            hidden_input = hidden
            x_and_hidden = torch.cat([x_input[:,:4*Nt], hidden_input],dim=1)
            hidden = self.tanh(self.conv(x = x_and_hidden, 
                     edge_index = edge_index[:,t,:], edge_attr = edge_attr[:,t,:]))
            output = self.h2o(hidden)
            # Normalization of data beamformer due to power constraint
            nor = torch.sqrt(torch.sum(torch.mul(output,output),axis=1))
            nor = nor.unsqueeze(axis=-1)
            comp1 = torch.ones(output.size(), device=device)
            comb = torch.div(output,torch.max(comp1,nor) )
            outputs[t,:,:] = comb
        return outputs     

def loss_function(data, out, K):
    loss0 = 0
    for i in range(frames-1):
        # In PG4U, the output data beamformer in the current frame will be used in the next frame
        H1 = data.y[:,i+1,:,:,:Nt]
        H2 = data.y[:,i+1,:,:,Nt:]
        p1 = out[i,:,:Nt]
        p2 = out[i,:,Nt:]
        p1 = torch.reshape(p1,(-1,K,1,Nt))
        p2 = torch.reshape(p2,(-1,K,1,Nt))
        rx_power1 = torch.mul(H1, p1)
        rx_power1 = torch.sum(rx_power1,axis=-1)
        rx_power2 = torch.mul(H2, p2)
        rx_power2 = torch.sum(rx_power2,axis=-1)
        rx_power3 = torch.mul(H1, p2)
        rx_power3 = torch.sum(rx_power3,axis=-1)
        rx_power4 = torch.mul(H2, p1)
        rx_power4 = torch.sum(rx_power4,axis=-1)

        rx_power = torch.mul(rx_power1 + rx_power2,rx_power1 + rx_power2) + torch.mul(rx_power3 - rx_power4,rx_power3 - rx_power4)
        mask = torch.eye(K, device = device)
        valid_rx_power = torch.sum(torch.mul(rx_power, mask), 1)
        interference = torch.sum(torch.mul(rx_power, 1 - mask), 1) + var
        # Designed loss function in (34)
        rate = torch.log(1 + torch.div(valid_rx_power, interference))
        minrate = torch.min(rate, axis=1)[0]
        loss0 += torch.neg(torch.mean(torch.mul(minrate,minrate)))
        
        # # One can also try the utility-based loss function below
        # sinr = torch.div(valid_rx_power, interference)
        # a = torch.neg(torch.mul(packet_length,np.log(2))) + torch.mul(frame_symbols,torch.log(1+sinr))
        # b = torch.div(a,np.sqrt(frame_symbols))
        # reliability = torch.special.erfc(b/np.sqrt(2))/2
        # r_max = torch.max(reliability,axis=1)[0]
        # loss0 += torch.mean(torch.log10(1e-5+r_max)+5)
    loss = torch.div(loss0, frames-1)
    return loss

def loss_and_QoS_evaluation(data, out, K):
    loss = 0
    QoS = 0
    for i in range(frames-1):
        # In PG4U, the output data beamformer in the current frame will be used in the next frame
        H1 = data.y[:,i+1,:,:,:Nt]
        H2 = data.y[:,i+1,:,:,Nt:]
        p1 = out[i,:,:Nt]
        p2 = out[i,:,Nt:]
        p1 = torch.reshape(p1,(-1,K,1,Nt))
        p2 = torch.reshape(p2,(-1,K,1,Nt))
        rx_power1 = torch.mul(H1, p1)
        rx_power1 = torch.sum(rx_power1,axis=-1)
        rx_power2 = torch.mul(H2, p2)
        rx_power2 = torch.sum(rx_power2,axis=-1)
        rx_power3 = torch.mul(H1, p2)
        rx_power3 = torch.sum(rx_power3,axis=-1)
        rx_power4 = torch.mul(H2, p1)
        rx_power4 = torch.sum(rx_power4,axis=-1)
        rx_power = torch.mul(rx_power1 + rx_power2,rx_power1 + rx_power2) + torch.mul(rx_power3 - rx_power4,rx_power3 - rx_power4)
        mask = torch.eye(K, device = device)
        valid_rx_power = torch.sum(torch.mul(rx_power, mask), 1)
        interference = torch.sum(torch.mul(rx_power, 1 - mask), 1) + var
        sinr = torch.div(valid_rx_power, interference)

        # Evaluation of utility loss
        a1 = -torch.mul(packet_length,np.log(2))+torch.mul(frame_symbols,torch.log(1+sinr))
        b1 = torch.div(a1,torch.sqrt(torch.mul(frame_symbols,1-torch.pow(1+sinr,-2))))
        reliability = torch.special.erfc(b1/np.sqrt(2))/2
        r_max = torch.max(reliability, axis=1)[0]
        loss += torch.mean(torch.log10(1e-5+r_max)+5)
        # Evaluation of QoS outage probability
        outage_index = torch.tensor(r_max > 1e-5, dtype=torch.float)
        QoS += torch.mean(outage_index)
    loss_output = torch.div(loss, frames-1)
    QoS_output = torch.div(QoS, frames-1)
    return loss_output, QoS_output

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(data,out,links)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / train_layouts

def test():
    model.eval()
    total_loss = 0
    total_QoS = 0
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            loss, QoS = loss_and_QoS_evaluation(data,out,links)
            total_loss += loss.item() * data.num_graphs
            total_QoS += QoS.item() * data.num_graphs
    return total_loss / test_layouts, total_QoS / test_layouts

train_K = 20
test_K = 20
# To train a better model and evluate the QoS accurately,
# one may increase the layouts to 20000 and 50000
train_layouts = 2000
test_layouts = 2000
train_config = init_parameters()
test_config = init_parameters()
var = train_config.output_noise_power / train_config.tx_power
# For bandwidth 5MHz and frame duration 1ms,
# there is a total number of 5000 symbols per frame
total_frame_symbols = 5000
packet_length = 128
graph_embedding_size = 16
# Channel estimation overhead and message passing overhead,
# 1 symbol (0.2 microsecond for bandwidth 5MHz)
O_csi = 1
O_mp = 1
# Computation delay 500 symbols (100 microseconds)
O_delay = 500
Nt = 4
frame_num = 10

print('Data generation')
# Data generation
# Train data
layouts, train_dists = wg.generate_layouts(train_config, train_layouts) 
train_path_losses = wg.compute_path_losses(train_config, train_dists)
train_path_losses = helper_functions.add_shadowing(train_path_losses)
train_csis = helper_functions.generate_csis(frame_num, train_path_losses,Nt)

# Test data
layouts, test_dists = wg.generate_layouts(test_config, test_layouts)
test_path_losses = wg.compute_path_losses(test_config, test_dists)
test_path_losses = helper_functions.add_shadowing(test_path_losses)
test_csis = helper_functions.generate_csis(frame_num, test_path_losses,Nt)

# Remaining frame symbols for PG4U by considering 
# overhead for CSI estimation and message passing
pg4u_frame_symbols = max(0, total_frame_symbols - train_K*Nt*O_csi)

# Data normalization
# Normalization of directlink CSIs
train_csi_real, train_csi_imag = np.real(train_csis), np.imag(train_csis)
test_csi_real, test_csi_imag = np.real(test_csis), np.imag(test_csis)
train_directlink_csis_real = get_directLink_csi(train_csi_real)
train_directlink_csis_imag = get_directLink_csi(train_csi_imag)
test_directlink_csis_real = get_directLink_csi(test_csi_real)
test_directlink_csis_imag = get_directLink_csi(test_csi_imag)
norm_train_directlink_real, norm_test_directlink_real = normalize_directlink_data(train_directlink_csis_real, test_directlink_csis_real)
norm_train_directlink_imag, norm_test_directlink_imag = normalize_directlink_data(train_directlink_csis_imag, test_directlink_csis_imag)

#Normalization constants for aggregation over-the-air
norm_aggOTA_mean, norm_aggOTA_var = normalize_agg_constants(train_csis)

# Graph data processing
print('Graph data processing')
train_data_list = proc_data(train_csis, norm_train_directlink_real,norm_train_directlink_imag,train_K)
test_data_list = proc_data(test_csis, norm_test_directlink_real,norm_test_directlink_imag, test_K)

# Train of PG4U
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PG4U().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
train_loader = DataLoader(train_data_list, batch_size=100, shuffle=True,num_workers=0)
test_loader = DataLoader(test_data_list, batch_size=100, shuffle=False, num_workers=0)
for epoch in range(1,51):
    batches = 100
    frames = frame_num
    links = train_K
    frame_symbols = pg4u_frame_symbols
    train_loss = train()
    test_loss, test_QoE = test()
    print('Epoch {:03d}, Train Loss: {:.4f}, Test Loss: {:.4f}, Test QoE: {:.4f}'.format(
            epoch, train_loss, test_loss, test_QoE))
    scheduler.step()

#Test for scalability and various system parameters, an example
gen_tests = [10, 15, 20, 25, 30, 35]
packet_length = 128
Nt = 4
O_csi = 1
O_mp = 1
O_processing = 500
var = train_config.output_noise_power / train_config.tx_power
test_K = 20
total_frame_symbols = 5000
frame_num = 10
density = train_config.field_length**2/train_K

for test_K in gen_tests:
    print('<<<<<<<<<<<<<< Num of Links is {:03d} >>>>>>>>>>>>>:'.format(test_K))
    # Generate test data
    test_config.n_links = test_K
    field_length = int(np.sqrt(density*test_K))
    test_config.field_length = field_length
    layouts, test_dists = wg.generate_layouts(test_config, test_layouts)
    test_path_losses = wg.compute_path_losses(test_config, test_dists)
    test_path_losses = helper_functions.add_shadowing(test_path_losses)
    test_csis = helper_functions.generate_csis(frame_num,test_path_losses,Nt)

    pg4u_frame_symbols = max(0, total_frame_symbols - test_K*Nt*O_csi)

    test_csi_real, test_csi_imag = np.real(test_csis), np.imag(test_csis)
    test_directlink_csis_real = get_directLink_csi(test_csi_real)
    test_directlink_csis_imag = get_directLink_csi(test_csi_imag)
    norm_train_directlink_real, norm_test_directlink_real = normalize_directlink_data(train_directlink_csis_real, test_directlink_csis_real)
    norm_train_directlink_imag, norm_test_directlink_imag = normalize_directlink_data(train_directlink_csis_imag, test_directlink_csis_imag)
    
    test_data_list = proc_data(test_csis, norm_test_directlink_real,norm_test_directlink_imag, test_K)
    test_loader = DataLoader(test_data_list, batch_size=100, shuffle=False, num_workers=0)
    batches = 100
    frames = frame_num
    links = test_K
    frame_symbols = pg4u_frame_symbols
    test_loss, test_QoE = test()
    print('GNN Loss: {:.4f} and QoE: {:.4f}:',test_loss, test_QoE)
