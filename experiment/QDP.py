import torch
import torch.distributions as distributions
import copy

def Quantization(param_client_data, param_client_model_k_data, Ran, task):
    Delta_data = param_client_data.to(task.device) - param_client_model_k_data.to(task.device)
    original_shape = Delta_data.shape
    Delta_data_1d = Delta_data.view(-1)
    Ran_new = Ran * (2** task.b - 1)/(2**task.b-1-2 * task.alpha).to(task.device)
    Q_ruler = torch.linspace(-Ran_new, Ran_new, 2**task.b).to(task.device)
    Q_step = (Q_ruler[1] - Q_ruler[0]).to(task.device) 
    Delta_data_1d = clip_Q(Delta_data_1d, Ran, Q_step)
    h = best_Q(Delta_data_1d, Ran_new, Q_step) 
    Delta_data_1d = dp_sample(Delta_data_1d, h, Q_ruler, task)
    Delta_data_new = Delta_data_1d.view(original_shape).to(task.device)
    Q_param = param_client_model_k_data + Delta_data_new
    return Q_param

def SQ_sample(data, h, Q_ruler, task):
    probs = torch.cat(((Q_ruler[h+1] - data).view(-1,1),(data - Q_ruler[h]).view(-1,1)),1)
    probs = probs / probs.sum(dim=1, keepdim=True).to(task.device)
    probs = probs.clamp(min=1e-15)  
    dis = distributions.Categorical(probs=probs)
    values = torch.cat((Q_ruler[h].view(-1,1), Q_ruler[h+1].view(-1,1)),1)
    index = dis.sample((1,))
    return values[torch.arange(values.size()[0]),index[0]]

def best_Q(data, Ran, Q_step):
    h = torch.floor((data + Ran)/Q_step).to(torch.int64)
    return h

def clip_Q(data, Ran, Q_step):
    data.clamp_(-Ran + 0.5 * Q_step, Ran - 0.5 * Q_step)
    return data

def dp_sample(data, h, Q_ruler, task): 
    task.dp_times += len(h)
    var_dis_2(h, task) 
    probs_L,probs_R = Laplace_or_Gauss_sample(h, Q_ruler, task)
    probs_L = probs_L.clamp(min=1e-15) 
    probs_R = probs_R.clamp(min=1e-15) 
    dis_l = distributions.Categorical(probs=probs_L)
    dis_r = distributions.Categorical(probs=probs_R)
    
    values_L = Q_ruler.repeat(len(data),1).to(task.device)
    values_R = Q_ruler.repeat(len(data),1).to(task.device)
    index_L = dis_l.sample((1,))
    index_R = dis_r.sample((1,))
    q_l = values_L[torch.arange(values_L.size()[0]),index_L[0]].to(task.device)
    q_r = values_R[torch.arange(values_R.size()[0]),index_R[0]].to(task.device)
    probs_F = torch.cat(((q_r - data).view(-1,1), (data - q_l).view(-1,1)),1).to(task.device)
    probs_F = probs_F / probs_F.sum(dim=1, keepdim=True).to(task.device)
    probs_F = probs_F.clamp(min=1e-15)  
    
    dis_f = distributions.Categorical(probs=probs_F)
    values = torch.cat((q_l.view(-1,1), q_r.view(-1,1)),1).to(task.device)
    index = dis_f.sample((1,)).to(task.device)
    return values[torch.arange(values.size()[0]),index[0]]

def Laplace_or_Gauss_sample(h, Q_ruler, task): 
    sigma_L = task.sigma_L.unsqueeze(1)
    L_ruler = Q_ruler.repeat(len(h),1)
    col_indices_L = torch.arange(L_ruler.size(1)).to(task.device)
    mask_L = col_indices_L <= h.unsqueeze(1).to(task.device)
    probs_L = torch.where(mask_L, L_ruler, torch.tensor(0.0))
    nth_values_L = torch.gather(probs_L, 1, (h).unsqueeze(1))
    update_mask_L = mask_L & (col_indices_L < h.unsqueeze(1))
    final_L = torch.where(update_mask_L, nth_values_L, probs_L)
    new_L = torch.where(mask_L, torch.exp(-abs(probs_L - final_L)/sigma_L), torch.tensor(0.0))
    probs_L = new_L/new_L.sum(dim=1, keepdim=True)
    sigma_R = task.sigma_R.unsqueeze(1)
    R_ruler = Q_ruler.repeat(len(h),1)
    col_indices_R = torch.arange(R_ruler.size(1))
    mask_R = col_indices_R.to(task.device) > h.unsqueeze(1).to(task.device)
    probs_R = torch.where(mask_R, R_ruler, torch.tensor(0.0))
    nth_values_R = torch.gather(probs_R, 1, (h+1).unsqueeze(1))
    update_mask_R = mask_R.to(task.device) & (col_indices_R.to(task.device) >= h.unsqueeze(1).to(task.device))
    final_R = torch.where(update_mask_R, nth_values_R, probs_R)
    if task.opt == 'ALQ':
        new_R = torch.where(mask_R, torch.exp(-abs(probs_R - final_R)/sigma_R), torch.tensor(0.0))
    elif task.opt == 'GSQ':
        new_R = torch.where(mask_R, torch.exp(-(probs_R - final_R)**2/(2*sigma_R**2)), torch.tensor(0.0))
    probs_R = new_R/new_R.sum(dim=1, keepdim=True)
    return probs_L.to(task.device), probs_R.to(task.device)

def var_dis_2(h, task): 
    if task.opt == 'ALQ':
        rate = ((h+1)/(2**task.b-h-1))**task.hyper 
        eps_re = task.eps_step - torch.log((2**task.b-task.alpha)*(2**task.b-task.alpha)/task.alpha/(task.alpha+1))
        if eps_re < 0:
            raise ValueError("Every iteration of eps is not enough.")
        else:
             task.sigma_R = ((rate + 1) * 2**task.b - 3*rate - 1) / (rate * eps_re)
             task.sigma_L = rate * task.sigma_R



    