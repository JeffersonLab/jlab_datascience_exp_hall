import torch
import logging

class PunziNetLoss(object):
    '''
    Define the punzi-net loss, according to this paper: https://arxiv.org/pdf/2110.00810.pdf

    Most of the code used here is taken from: https://github.com/feichtip/punzinet/blob/main/src/punzinet/fom.py
    '''

    # Initialize:
    #*********************
    def __init__(self,n_mass_hypotheses,a=3.0,b=1.28155,scale=1.0,torch_device="cpu"):
        self.n_mass_hypotheses = n_mass_hypotheses

        assert self.n_mass_hypotheses >0, logging.error(f">>> PunziLoss: You did not provide a positive number of mass hypotheses: n_mass_hypotheses = {self.n_mass_hypotheses} <<<")
        
        self.a = a
        self.b = b
        self.scale = scale

        self.torch_device = torch_device
    #*********************

    # Compute sparse matrices --> This will be important for calculating the punzi fom:
    # (I honestly have no clue what is going on here, so I will blindly utilize the code from here: https://github.com/feichtip/punzinet/blob/main/src/punzinet/fom.py
    #*********************
    def compute_sparse_matrices(self,sigma_components):
        sig_m_range = sigma_components[:,0]
        gen_mass = sigma_components[:,1]
        range_idx_low = sigma_components[:,2]
        range_idx_high = sigma_components[:,3]
        
        bkg = gen_mass == -999
        sig_m_range = sig_m_range.type(torch.BoolTensor)
        sig_m_range.to(self.torch_device)

        idx_range = (1 + range_idx_high - range_idx_low).byte()
        idx_range.to(self.torch_device)
        v1, v2, v1_bkg, v2_bkg = ([], [], [], [])
        
        #+++++++++++++++++++++++++++++++++++++++++++++++
        for j in range(idx_range.max()):
           indices = torch.nonzero(idx_range > j)[:, 0]
           indices.to(self.torch_device)
           v1.append(range_idx_low[indices].to(self.torch_device) + j)
           v2.append(indices)

           indices_bkg = torch.nonzero((idx_range > j) & bkg)[:, 0]
           indices_bkg.to(self.torch_device)
           v1_bkg.append(range_idx_low[indices_bkg].to(self.torch_device) + j)
           v2_bkg.append(indices_bkg)
        #+++++++++++++++++++++++++++++++++++++++++++++++

        i = torch.zeros(2, idx_range.sum(), dtype=torch.long,device=self.torch_device)
        i[0, :] = torch.cat(v1).to(self.torch_device)
        i[1, :] = torch.cat(v2).to(self.torch_device)
        v = torch.ByteTensor([1]).expand(i.shape[1]).to(self.torch_device)

        _, inverse_indices = torch.unique(gen_mass[sig_m_range], sorted=True, return_inverse=True)
        inverse_indices.to(self.torch_device)
        i_sig = torch.zeros(2, len(inverse_indices), dtype=torch.long,device=self.torch_device)
        i_sig[0, :] = inverse_indices
        i_sig[1, :] = torch.arange(len(gen_mass))[sig_m_range].to(self.torch_device)
        v_sig = torch.ByteTensor([1]).expand(i_sig.shape[1]).to(self.torch_device)

        i_bkg = torch.zeros(2, (idx_range * bkg).sum(), dtype=torch.long,device=self.torch_device)
        i_bkg[0, :] = torch.cat(v1_bkg).to(self.torch_device)
        i_bkg[1, :] = torch.cat(v2_bkg).to(self.torch_device)
        v_bkg = torch.ByteTensor([1]).expand(i_bkg.shape[1]).to(self.torch_device)

        sparse_shape = torch.Size([self.n_mass_hypotheses, len(gen_mass)])
        in_range = torch.sparse.ByteTensor(i, v, sparse_shape).to(self.torch_device)
        sig_sparse = torch.sparse.ByteTensor(i_sig, v_sig, sparse_shape).to(self.torch_device) * in_range
        bkg_sparse = torch.sparse.ByteTensor(i_bkg, v_bkg, sparse_shape).to(self.torch_device)

        return sig_sparse.to(self.torch_device), bkg_sparse.to(self.torch_device)
    #*********************

    # Get the punzi-loss itself:
    #*********************
    # Get the sensitivity first:
    def calc_punzi_sensitivity(self,signal_efficiency,background_events,lumi):
        signal_efficiency[signal_efficiency == 0.0] = 1E-6
        background_events[background_events == 0.0] = 1E-4
        
        sigma_norm = signal_efficiency * lumi
        sigma_raw = (self.a)**2 / 8 + 9 * (self.b)**2 / 13 + (self.a) * torch.sqrt(background_events) + (self.b) * torch.sqrt((self.b)**2 + 4 * (self.a) * torch.sqrt(background_events) + 4 * background_events) / 2
        
        return sigma_raw / sigma_norm
    #---------------------------

    # Compute the loss:
    def compute(self,sigma_components,network_response,weights,n_gen_signal,target_lumi):
        # Get signal / background matrices:
        sig_sparse_M, bkg_sparse_M = self.compute_sparse_matrices(sigma_components)

        # Calculate the signal efficiency:
        signal_efficiency = ((sig_sparse_M).float() @ network_response.reshape(-1, 1) * self.scaling / n_gen_signal).flatten()
        # Determine background events:
        background_events = ((bkg_sparse_M).float() @ (network_response * weights).reshape(-1, 1) * self.scaling).flatten()
        return self.calc_punzi_sensitivity(signal_efficiency,background_events,target_lumi)
    #*********************