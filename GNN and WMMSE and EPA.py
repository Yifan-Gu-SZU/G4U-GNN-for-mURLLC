import scipy.io as sio                     
import numpy as np                         
import matplotlib.pyplot as plt           
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, Tanh, BatchNorm1d as BN
import wireless_networks_generator as wg
import helper_functions

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

def normalize_data(train_data, test_data):
    # Normlize train direct link
    tmp_mask = np.expand_dims(np.eye(train_K),axis=-1)
    tmp_mask = [tmp_mask for i in range(Nt)]
    mask = np.concatenate(tmp_mask,axis=-1)
    mask = np.expand_dims(mask,axis=0)
    train_copy = np.copy(train_data)
    diag_H = np.multiply(mask,train_copy)
    diag_mean = np.sum(diag_H/Nt)/train_layouts/train_K
    diag_var = np.sqrt(np.sum(np.square(diag_H-diag_mean))/train_layouts/train_K/Nt)
    tmp_diag = (diag_H - diag_mean)/diag_var

    # Normlize train interference link
    off_diag = train_copy - diag_H
    off_diag_mean = np.sum(off_diag/Nt)/train_layouts/train_K/(train_K-1)
    off_diag_var = np.sqrt(np.sum(np.square(off_diag-off_diag_mean))/Nt/train_layouts/train_K/(train_K-1))
    tmp_off = (off_diag - off_diag_mean)/off_diag_var
    tmp_off_diag = tmp_off - np.multiply(tmp_off,mask) 
    norm_train = np.multiply(tmp_diag,mask) + tmp_off_diag
    
    # Normlize test data
    tmp_mask = np.expand_dims(np.eye(test_K),axis=-1)
    tmp_mask = [tmp_mask for i in range(Nt)]
    mask = np.concatenate(tmp_mask,axis=-1)
    mask = np.expand_dims(mask,axis=0)
    
    test_copy = np.copy(test_data)
    diag_H = np.multiply(mask,test_copy)
    tmp_diag = (diag_H - diag_mean)/diag_var
    
    off_diag = test_copy - diag_H
    tmp_off = (off_diag - off_diag_mean)/off_diag_var
    tmp_off_diag = tmp_off - np.multiply(tmp_off,mask)
    
    norm_test = np.multiply(tmp_diag,mask) + tmp_off_diag
    return norm_train, norm_test

def build_graph(CSI, norm_csi_real, norm_csi_imag, K):
    n = CSI.shape[0]
    Nt = CSI.shape[2]
    x1 = np.array([norm_csi_real[ii,ii,:] for ii in range(K)])
    x2 = np.array([norm_csi_imag[ii,ii,:] for ii in range(K)])
    x3 = 1/np.sqrt(Nt)*np.zeros((n,graph_embedding_size))
    x = np.concatenate((x3,x1,x2),axis=1)
    x = torch.tensor(x, dtype=torch.float)

   # Consider fully connected graph 
    dist2 = np.copy(norm_csi_real[:,:,0])
    mask = np.eye(K)
    diag_dist = np.multiply(mask,dist2)
    dist2 = dist2 - diag_dist
    attr_ind = np.nonzero(dist2)  
    edge_attr_real = norm_csi_real[attr_ind]
    edge_attr_imag = norm_csi_imag[attr_ind]
    edge_attr = np.concatenate((edge_attr_real,edge_attr_imag), axis=1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    attr_ind = np.array(attr_ind)
    adj = np.zeros(attr_ind.shape)
    adj[0,:] = attr_ind[1,:]
    adj[1,:] = attr_ind[0,:]
    edge_index = torch.tensor(adj, dtype=torch.long)
    
    H1 = np.expand_dims(np.real(CSI),axis=-1)
    H2 = np.expand_dims(np.imag(CSI),axis=-1)
    HH = np.concatenate((H1,H2),axis=-1)
    y = torch.tensor(np.expand_dims(HH,axis=0), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.contiguous(),edge_attr = edge_attr, y = y)
    return data

def proc_data(HH, norm_csi_real, norm_csi_imag, K):
    n = HH.shape[0]
    data_list = []
    for i in range(n):
        data = build_graph(HH[i,:,:,:], norm_csi_real[i,:,:,:], norm_csi_imag[i,:,:,:], K)
        data_list.append(data)
    return data_list

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels))
    ]) 
  
class GNNconv(MessagePassing):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(GNNconv, self).__init__(aggr='max', **kwargs)
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        
    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)
        comb = self.mlp2(tmp)
        return torch.cat([comb,x[:,graph_embedding_size:]],dim=1)
        
    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp)
        return agg

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1,self.mlp2)

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()

        self.mlp1 = MLP([4*Nt+graph_embedding_size, 32, 32])
        self.mlp2 = MLP([32+2*Nt+graph_embedding_size, 32, graph_embedding_size])
        self.conv = GNNconv(self.mlp1,self.mlp2)
        self.h2o = MLP([graph_embedding_size, 32])
        self.h2o = Seq(*[self.h2o,Seq(Lin(32, 2*Nt, bias = True), Tanh())])

    def forward(self, data):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x1 = self.conv(x = x0, edge_index = edge_index, edge_attr = edge_attr)
        #x2 = self.conv(x = x1, edge_index = edge_index, edge_attr = edge_attr)
        out = self.conv(x = x1, edge_index = edge_index, edge_attr = edge_attr)
        output = self.h2o(out[:,:graph_embedding_size])
        # Normalization due to maximum power constraint
        nor = torch.sqrt(torch.sum(torch.mul(output,output),axis=1))
        nor = nor.unsqueeze(axis=-1)
        comp1 = torch.ones(output.size(), device=device)
        norm_output = torch.div(output,torch.max(comp1,nor) )
        return norm_output

def loss_function(data,p,K,N):
    H1 = data.y[:,:,:,:,0]
    H2 = data.y[:,:,:,:,1]
    p1 = p[:,:N]
    p2 = p[:,N:2*N]
    p1 = torch.reshape(p1,(-1,K,1,N))
    p2 = torch.reshape(p2,(-1,K,1,N))
    
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
    valid_rx_power = torch.sum(torch.mul(rx_power, mask), axis=1)
    interference = torch.sum(torch.mul(rx_power, 1 - mask), axis=1) + var
    # Designed loss function in (34)
    rate = torch.log(1 + torch.div(valid_rx_power, interference))
    minrate = torch.min(rate, axis=1)[0]
    loss = torch.mean(torch.mul(minrate,minrate))
    loss = torch.neg(loss)
    
    # # One can also try the utility-based loss function below
    # sinr = torch.div(valid_rx_power, interference)
    # a = -torch.mul(packet_length,np.log(2))+torch.mul(frame_symbols,torch.log(1+sinr))
    # b = torch.div(a,np.sqrt(frame_symbols))
    # reliability = torch.special.erfc(b/np.sqrt(2))/2
    # r_max = torch.max(reliability, axis=1)[0]
    # loss = torch.mean(torch.log10(1e-5+r_max)+5)
    return loss

def loss_and_QoS_evaluation(data,p,K,N):
    H1 = data.y[:,:,:,:,0]
    H2 = data.y[:,:,:,:,1]
    p1 = p[:,:N]
    p2 = p[:,N:2*N]
    p1 = torch.reshape(p1,(-1,K,1,N))
    p2 = torch.reshape(p2,(-1,K,1,N))
    
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
    valid_rx_power = torch.sum(torch.mul(rx_power, mask), axis=1)
    interference = torch.sum(torch.mul(rx_power, 1 - mask), axis=1) + var
    sinr = torch.div(valid_rx_power, interference)

    # Evaluation of utility loss
    a1 = -torch.mul(packet_length,np.log(2))+torch.mul(frame_symbols,torch.log(1+sinr))
    b1 = torch.div(a1,torch.sqrt(torch.mul(frame_symbols,1-torch.pow(1+sinr,-2))))
    reliability = torch.special.erfc(b1/np.sqrt(2))/2
    r_max = torch.max(reliability, axis=1)[0]
    loss = torch.mean(torch.log10(1e-5+r_max)+5)
    # Evaluation of QoS outage probability
    index = torch.tensor(r_max > 1e-5, dtype=torch.float)
    QoS = torch.mean(index)
    return loss, QoS

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(data,out,train_K,Nt)
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
            loss, QoS = loss_and_QoS_evaluation(data,out,test_K,Nt)
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
# Since temporal correlation is not considered in GNN,
# we only consider 1 frame for training and testing
train_csis = train_csis[:,1,:,:]

# Test data
layouts, test_dists = wg.generate_layouts(test_config, test_layouts)
test_path_losses = wg.compute_path_losses(test_config, test_dists)
test_path_losses = helper_functions.add_shadowing(test_path_losses)
test_csis = helper_functions.generate_csis(frame_num, test_path_losses,Nt)
test_csis = test_csis[:,1,:,:]

# Remaining frame symbols for data transmission for gnn
# One can set gnn_frame_symbols to total_frame_symbols for the case without considering any overhead
gnn_frame_symbols = max(0,total_frame_symbols - train_K*train_K*Nt*O_csi-2*train_K*(train_K-1)*O_mp-5*O_delay)

# Data normalization
train_csi_real, train_csi_imag = np.real(train_csis), np.imag(train_csis)
test_csi_real, test_csi_imag = np.real(test_csis), np.imag(test_csis)
norm_train_real, norm_test_real = normalize_data(train_csi_real,test_csi_real)
norm_train_imag, norm_test_imag = normalize_data(train_csi_imag,test_csi_imag)

# Graph data processing
print('Graph data processing')
train_data_list = proc_data(train_csis, norm_train_real, norm_train_imag, train_K)
test_data_list = proc_data(test_csis, norm_test_real, norm_test_imag,  test_K)

print('WMMSE and EPA computation')
# WMMSE
Y = helper_functions.batch_wmmse(test_csis.transpose(0,2,1,3),var,Nt, test_K)
wmmse_loss, wmmse_qoe = helper_functions.loss_and_QoS( test_csis,Y,var,packet_length, total_frame_symbols)
print('WMMSE loss and QoE outage probability:',wmmse_loss, wmmse_qoe)

# EPA
epa_p = 1/np.sqrt(Nt)*np.ones((test_K,Nt),dtype=complex)
epa_loss, epa_qoe = helper_functions.loss_and_QoS( test_csis,epa_p,var,packet_length, total_frame_symbols)
print('EPA loss and QoE outage probability:',epa_loss, epa_qoe)

#Train of GNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)


train_loader = DataLoader(train_data_list, batch_size=100, shuffle=True,num_workers=0)
test_loader = DataLoader(test_data_list, batch_size=100, shuffle=False, num_workers=0)

for epoch in range(1,51):
    frame_symbols = gnn_frame_symbols
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
    test_csis = test_csis[:,1,:,:]

    gnn_frame_symbols = max(0,total_frame_symbols - test_K*test_K*Nt*O_csi-2*test_K*(test_K-1)*O_mp-5*O_delay)

    train_csi_real, train_csi_imag = np.real(train_csis), np.imag(train_csis)
    test_csi_real, test_csi_imag = np.real(test_csis), np.imag(test_csis)
    norm_train_real, norm_test_real = normalize_data(train_csi_real,test_csi_real)
    norm_train_imag, norm_test_imag = normalize_data(train_csi_imag,test_csi_imag)
    
    # Test for WMMSE
    Y = helper_functions.batch_wmmse(test_csis.transpose(0,2,1,3),var,Nt, test_K)
    wmmse_loss, wmmse_qoe = helper_functions.loss_and_QoS( test_csis,Y,var,packet_length, total_frame_symbols)
    print('WMMSE loss and QoE outage probability:',wmmse_loss, wmmse_qoe)
    
    # Test for EPA
    epa_p = 1/np.sqrt(Nt)*np.ones((test_K,Nt),dtype=complex)
    epa_loss, epa_qoe = helper_functions.loss_and_QoS( test_csis,epa_p,var,packet_length, total_frame_symbols)
    print('EPA loss and QoE outage probability:',epa_loss, epa_qoe)

    test_data_list = proc_data(test_csis, norm_test_real, norm_test_imag,  test_K)

    test_loader = DataLoader(test_data_list, batch_size=100, shuffle=False, num_workers=0)
    frame_symbols = gnn_frame_symbols
    test_loss, test_QoE = test()
    print('GNN Loss: {:.4f} and QoE: {:.4f}:',test_loss, test_QoE)

