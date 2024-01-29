import numpy as np
import scipy as sp
from scipy import special

def layout_generate(general_para):
    while(True):
        N = general_para.n_links
        # first, generate transmitters' coordinates
        tx_xs = np.random.uniform(low=0, high=general_para.field_length, size=[N,1])
        tx_ys = np.random.uniform(low=0, high=general_para.field_length, size=[N,1])
        # generate rx one by one rather than N together to ensure checking validity one by one
        rx_xs = []; rx_ys = []
        for i in range(N):
            got_valid_rx = False
            while(not got_valid_rx):
                pair_dist = np.random.uniform(low=general_para.shortest_directLink_length, high=general_para.longest_directLink_length)
                pair_angles = np.random.uniform(low=0, high=np.pi*2)
                rx_x = tx_xs[i] + pair_dist * np.cos(pair_angles)
                rx_y = tx_ys[i] + pair_dist * np.sin(pair_angles)
                if(0<=rx_x<=general_para.field_length and 0<=rx_y<=general_para.field_length):
                    got_valid_rx = True
            rx_xs.append(rx_x); rx_ys.append(rx_y)
        # For now, assuming equal weights and equal power, so not generating them
        layout = np.concatenate((tx_xs, tx_ys, rx_xs, rx_ys), axis=1)
        distances = np.zeros([N, N])
        # Compute distance between every possible Tx/Rx pair
        for rx_index in range(N):
            for tx_index in range(N):
                tx_coor = layout[tx_index][0:2]
                rx_coor = layout[rx_index][2:4]
                # According to paper notation convention, Hij is from jth transmitter to ith receiver
                distances[rx_index][tx_index] = np.linalg.norm(tx_coor - rx_coor)
        # Check whether link is too close
        dcopy = np.copy(distances)
        mask = np.eye(N)
        d_diag = np.multiply(mask,dcopy)
        d_offdiag = dcopy - d_diag
        d_check = d_diag*1000 + d_offdiag
        if(np.min(d_check)>general_para.shortest_crossLink_length):
            break
    return layout, distances

# Add shadowing into channel losses
def add_shadowing(channel_losses):
    shadow_coefficients = np.random.normal(loc=0, scale=3, size=np.shape(channel_losses))
    channel_losses = channel_losses * np.power(10.0, shadow_coefficients / 10)
    return channel_losses

# CSI generation according to the AR1 model
def generate_csis(frames, train_path_losses,Nt):
    n = np.shape(train_path_losses)
    n_links = np.multiply(n[1],n[2])
    csi_seq = np.zeros((n[0],frames,n[1],n[2],Nt), dtype=complex)
    for nt in range(Nt):
        for i in range(n[0]):
            # channel correlation coefficient
            r = 0.99
            alpha = np.resize(train_path_losses[i,:,:],n_links)
            noise_var = np.multiply(alpha,1-np.power(r,2))
            # channel coefficient matrix
            sims_real = np.zeros((frames,n_links))
            sims_imag = np.zeros((frames,n_links))
        # generate the channel coefficients for consecutive frames
            sims_real[0,:] = np.random.normal(loc = 0, scale = np.sqrt(alpha))
            sims_imag[0,:] = np.random.normal(loc = 0, scale = np.sqrt(alpha))
            for j in range(frames-1):
                sims_real[j+1,:] = np.multiply(r,sims_real[j,:]) + np.random.normal(loc = 0, scale = np.sqrt(noise_var))
                sims_imag[j+1,:] = np.multiply(r,sims_imag[j,:]) + np.random.normal(loc = 0, scale = np.sqrt(noise_var))
            layout_csi_seq = 1/np.sqrt(2)*(sims_real+1j*sims_imag)
            csi_seq[i,:,:,:,nt] = np.resize(layout_csi_seq,(frames,n[1],n[2]))
    return csi_seq

# Used for WMMSE
def np_WMMSE_vector(b_int, H, Pmax, var_noise):
    # fix transpose and conjudgate
    K = b_int.shape[0]
    N = b_int.shape[1]
    vnew = 0
    b = b_int
    f = np.zeros(K,dtype=complex)
    w = np.zeros(K,dtype=complex)

    mask = np.eye(K)

    btmp = np.reshape(b, (K,1,N))
    rx_power = np.multiply(H, b)
    rx_power = np.sum(rx_power,axis=-1)
    valid_rx_power = np.sum(np.multiply(rx_power, mask), axis=0)
    interference_rx = np.square(np.abs(rx_power))
    interference = np.sum(interference_rx, axis=1) + var_noise
    f = np.divide(valid_rx_power, interference)
    w = 1/(1 - valid_rx_power*f.conj())
    vnew = np.sum(np.log2(np.abs(w)))
    
    for iter in range(10):
        vold = vnew
        H_H = np.expand_dims(H.conj(),axis=-1)
        H_tmp = np.expand_dims(H,axis=-2)
        HH = np.matmul(H_H,H_tmp)

        UWU = np.reshape(w * (f.conj()).T * f,(K,1,1,1))
        btmp = np.sum(HH * UWU, axis=0)
        for ii in range(K):
            Hkk = np.expand_dims(H[ii,ii,:],axis=0)
            b[ii,:] = get_mu(Pmax,Hkk,btmp[ii,:,:],w[ii] * f[ii])

        btmp = np.reshape(b, (K,1,N))
        rx_power = np.multiply(H, b)
        rx_power = np.sum(rx_power,axis=-1)
        valid_rx_power = np.sum(np.multiply(rx_power, mask), axis=0)
        interference_rx = np.square(np.abs(rx_power))
        interference = np.sum(interference_rx, axis=1) + var_noise
        f = np.divide(valid_rx_power, interference)
        w = 1/(1 - valid_rx_power*f.conj())
        vnew = np.sum(np.log2(np.abs(w)))
        if abs(vnew - vold) <= 1e-3:
           break
    return b

# Used for WMMSE
def get_mu(Pmax,Hkk,btmp,wf):
    Lmu = 0
    N = Hkk.shape[1]

    I = np.eye(N)
    Hkk_H = (Hkk.conj()).T
    
    if(np.linalg.matrix_rank(btmp) == N 
        and np.linalg.norm(np.matmul(np.linalg.inv(btmp),Hkk_H ) * wf) < np.sqrt(Pmax)):
        return np.squeeze(np.matmul(np.linalg.inv(btmp),Hkk_H ) * wf)

    Lambda, D = np.linalg.eig(btmp)
    Lambda = np.diag(Lambda)
    HUW = Hkk_H*wf
    Phitmp = np.matmul(HUW,(HUW.conj()).T)
    DH = (D.conj()).T

    Phi = np.matmul(np.matmul(DH,Phitmp),D)
    Phimm = np.real(np.diag(Phi))
    Lambdamm = np.real(np.diag(Lambda))

    
    Rmu = 1
    Pcomp = np.sum(Phimm/(Lambdamm + Rmu)**2)
    while(Pcomp > Pmax):
        Rmu = Rmu*2
        Lmu = Rmu
        Pcomp = np.sum(Phimm/(Lambdamm + Rmu)**2)
    while(Rmu-Lmu > 1e-4):
        midmu = (Rmu + Lmu)/2
        Pcomp = np.sum(Phimm/(Lambdamm + midmu)**2)
        if(Pcomp < Pmax ):
            Rmu = midmu
        else: 
            Lmu = midmu
    ans = np.squeeze(np.matmul(np.linalg.inv(btmp + Rmu*I),Hkk_H ) * wf)

    return ans

# Used for WMMSE
def batch_wmmse(csis,var_noise, N_antenna, test_K):
    Nt = N_antenna
    K = test_K
    n = csis.shape[0]
    Y = np.zeros( (n,K,Nt),dtype=complex)
    Pini = 1/np.sqrt(Nt)*np.ones((K,Nt),dtype=complex)
    #Pini = np.ones((K,Nt),dtype=complex)
    for ii in range(n):
        Y[ii,:,:] = np_WMMSE_vector(np.copy(Pini), csis[ii,:,:,:], 1, var_noise)
    return Y

# Evaluate utility loss and QoS outage probability
def loss_and_QoS(H,p, var_noise, packet_length, frame_duration):
    K = H.shape[1]
    N = H.shape[-1]
    p = p.reshape((-1,K,1,N))
    rx_power = np.multiply(H, p)
    rx_power = np.sum(rx_power,axis=-1)
    rx_power = np.square(np.abs(rx_power))
    mask = np.eye(K)
    valid_rx_power = np.sum(np.multiply(rx_power, mask), axis=1)
    interference = np.sum(np.multiply(rx_power, 1 - mask), axis=1) + var_noise
    sinr = np.divide(valid_rx_power, interference)
    a = -packet_length*np.log(2)+frame_duration*np.log(1+sinr)
    v = 1 - 1 / np.power(1+sinr,2)
    b = a/np.sqrt(frame_duration*v)
    reliability = sp.special.erfc(b/np.sqrt(2))/2
    r_max = np.max(np.copy(reliability),axis = 1)
    return np.mean(np.log10(1e-5+r_max)+5), np.mean(np.int64(r_max>1e-5))