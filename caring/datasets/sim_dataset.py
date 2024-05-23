import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import ipdb as pdb

class StationaryDataset(Dataset):
    
    def __init__(self, directory, transition="noisecoupled_gaussian_ts_2lag"):
        super().__init__()
        self.path = os.path.join(directory, transition, "data.npz")
        self.npz = np.load(self.path)
        self.data = { }
        for key in ["yt", "xt"]:
            self.data[key] = self.npz[key]

    def __len__(self):
        return len(self.data["yt"])

    def __getitem__(self, idx):
        yt = torch.from_numpy(self.data["yt"][idx].astype('float32'))
        xt = torch.from_numpy(self.data["xt"][idx].astype('float32'))
        sample = {"yt": yt, "xt": xt}
        return sample

class TimeVaryingDataset(Dataset):
    
    def __init__(self, directory, transition="pnl_change_20", dataset="source"):
        super().__init__()
        self.path = os.path.join(directory, transition, dataset, "data.npz")
        self.npz = np.load(self.path)
        self.data = { }
        for key in ["yt", "xt", "ct"]:
            self.data[key] = self.npz[key]

    def __len__(self):
        return len(self.data["yt"])

    def __getitem__(self, idx):
        yt = torch.from_numpy(self.data["yt"][idx].astype('float32'))
        xt = torch.from_numpy(self.data["xt"][idx].astype('float32'))
        ct = torch.from_numpy(self.data["ct"][idx].astype('float32'))
        sample = {"yt": yt, "xt": xt, "ct": ct}
        return sample

class DANS(Dataset):
    def __init__(self, directory, dataset="da_10"):
        super().__init__()
        self.path = os.path.join(directory, dataset, "data.npz")
        self.npz = np.load(self.path)
        self.data = { }
        for key in ["y", "x", "c"]:
            self.data[key] = self.npz[key]

    def __len__(self):
        return len(self.data["y"])

    def __getitem__(self, idx):
        y = torch.from_numpy(self.data["y"][idx].astype('float32'))
        x = torch.from_numpy(self.data["x"][idx].astype('float32'))
        c = torch.from_numpy(self.data["c"][idx, None].astype('float32'))
        sample = {"y": y, "x": x, "c": c}
        return sample

class SimulationDatasetTSTwoSample(Dataset):
	
	def __init__(self, directory, transition="linear_nongaussian_ts"):
		super().__init__()
		self.path = os.path.join(directory, transition, "data.npz")
		self.npz = np.load(self.path)
		self.data = { }
		for key in ["yt", "xt"]:
			self.data[key] = self.npz[key]

	def __len__(self):
		return len(self.data["yt"])

	def __getitem__(self, idx):
		yt = torch.from_numpy(self.data["yt"][idx].astype('float32'))
		xt = torch.from_numpy(self.data["xt"][idx].astype('float32'))
		idx_rnd = random.randint(0, len(self.data["yt"])-1)
		ytr = torch.from_numpy(self.data["yt"][idx_rnd].astype('float32'))
		xtr = torch.from_numpy(self.data["xt"][idx_rnd].astype('float32'))
		sample = {"s1": {"yt": yt, "xt": xt},
				  "s2": {"yt": ytr, "xt": xtr}
				  }
		return sample


class SimulationDatasetTSTwoSampleNS(Dataset):
    
    def __init__(self, directory, transition="linear_nongaussian_ts", dataset='source'):
        super().__init__()
        self.path = os.path.join(directory, transition, dataset, "data.npz")
        self.npz = np.load(self.path)
        self.data = { }
        for key in ["yt", "xt", "ct"]:
            self.data[key] = self.npz[key]

    def __len__(self):
        return len(self.data["yt"])

    def __getitem__(self, idx):
        yt = torch.from_numpy(self.data["yt"][idx].astype('float32'))
        xt = torch.from_numpy(self.data["xt"][idx].astype('float32'))
        ct = torch.from_numpy(self.data["ct"][idx].astype('float32'))
        idx_rnd = random.randint(0, len(self.data["yt"])-1)
        ytr = torch.from_numpy(self.data["yt"][idx_rnd].astype('float32'))
        xtr = torch.from_numpy(self.data["xt"][idx_rnd].astype('float32'))
        ctr = torch.from_numpy(self.data["ct"][idx_rnd].astype('float32'))
        sample = {"s1": {"yt": yt, "xt": xt, "ct": ct},
                  "s2": {"yt": ytr, "xt": xtr, "ct": ctr}
                  }
        return sample

class SimulationDatasetPCL(Dataset):
    
    def __init__(self, directory, transition="linear_nongaussian_ts", lags=2):
        super().__init__()
        self.path = os.path.join(directory, transition, "data.npz")
        self.npz = np.load(self.path)
        self.L = lags
        self.data = { }
        for key in ["yt", "xt"]:
            self.data[key] = self.npz[key]

    def __len__(self):
        return len(self.data["yt"])

    def __getitem__(self, idx):
        yt = torch.from_numpy(self.data["yt"][idx].astype('float32'))
        xt = torch.from_numpy(self.data["xt"][idx].astype('float32'))
        xt_cur, xt_his = self.seq_to_pairs(xt)
        idx_rnd = random.randint(0, len(self.data["yt"])-1)
        ytr = torch.from_numpy(self.data["yt"][idx_rnd].astype('float32'))
        xtr = torch.from_numpy(self.data["xt"][idx_rnd].astype('float32'))
        xtr_cur, xtr_his = self.seq_to_pairs(xtr)
        xt_cat = torch.cat((xt_cur, xt_his), dim=1)
        xtr_cat = torch.cat((xt_cur, xtr_his), dim=1)

        sample = {"s1": {"yt": yt, "xt": xt},
                  "s2": {"yt": ytr, "xt": xtr},
                  "pos": {"x": xt_cat, "y": 1},
                  "neg": {"x": xtr_cat, "y": 0}
                  }
        return sample

    def seq_to_pairs(self, x):
        x = x.unfold(dimension = 0, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 1, 2)
        xx, yy = x[:,-1:], x[:,:-1]
        return xx, yy

class SimulationDatasetPCLNS(Dataset):
    
    def __init__(self, directory, transition="linear_nongaussian_ts", lags=2, dataset='source'):
        super().__init__()
        self.path = os.path.join(directory, transition, dataset, "data.npz")
        self.npz = np.load(self.path)
        self.L = lags
        self.data = { }
        for key in ["yt", "xt", "ct"]:
            self.data[key] = self.npz[key]

    def __len__(self):
        return len(self.data["yt"])

    def __getitem__(self, idx):
        yt = torch.from_numpy(self.data["yt"][idx].astype('float32'))
        xt = torch.from_numpy(self.data["xt"][idx].astype('float32'))
        ct = torch.from_numpy(self.data["ct"][idx].astype('float32'))
        xt_cur, xt_his = self.seq_to_pairs(xt)
        idx_rnd = random.randint(0, len(self.data["yt"])-1)
        ytr = torch.from_numpy(self.data["yt"][idx_rnd].astype('float32'))
        xtr = torch.from_numpy(self.data["xt"][idx_rnd].astype('float32'))
        ctr = torch.from_numpy(self.data["ct"][idx_rnd].astype('float32'))
        xtr_cur, xtr_his = self.seq_to_pairs(xtr)
        xt_cat = torch.cat((xt_cur, xt_his), dim=1)
        xtr_cat = torch.cat((xt_cur, xtr_his), dim=1)

        sample = {"s1": {"yt": yt, "xt": xt, "ct": ct},
                  "s2": {"yt": ytr, "xt": xtr, "ct": ctr},
                  "pos": {"x": xt_cat, "y": 1},
                  "neg": {"x": xtr_cat, "y": 0}
                  }
        return sample

    def seq_to_pairs(self, x):
        x = x.unfold(dimension = 0, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 1, 2)
        xx, yy = x[:,-1:], x[:,:-1]
        return xx, yy