import zipfile
import sqlite3
import time
import pandas as pd
import numpy as np
import os
import math
from typing import List, Optional, Dict
from datetime import timedelta, datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
from itertools import accumulate
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def power_forgetting_curve(t, s):
    return (1 + t / (9 * s)) ** -1

def next_interval(s, r):
    return np.maximum(1, np.round(9 * s * (1/r - 1)))

class FSRS(nn.Module):
    def __init__(self, w: List[float]):
        super(FSRS, self).__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))

    def stability_after_success(self, state: Tensor, new_d: Tensor, r: Tensor, rating: Tensor) -> Tensor:
        hard_penalty = torch.where(rating == 2, self.w[15], 1)
        easy_bonus = torch.where(rating == 4, self.w[16], 1)
        new_s = state[:,0] * (1 + torch.exp(self.w[8]) *
                        (11 - new_d) *
                        torch.pow(state[:,0], -self.w[9]) *
                        (torch.exp((1 - r) * self.w[10]) - 1) * 
                        hard_penalty * 
                        easy_bonus)
        return new_s

    def stability_after_failure(self, state: Tensor, new_d: Tensor, r: Tensor) -> Tensor:
        new_s = self.w[11] * \
                torch.pow(new_d, -self.w[12]) * \
                (torch.pow(state[:,0] + 1, self.w[13]) - 1) * \
                torch.exp((1 - r) * self.w[14])
        return new_s

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        '''
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 2], state[:,0] is stability, state[:,1] is difficulty
        :return state:
        '''
        if torch.equal(state, torch.zeros_like(state)):
            keys = torch.tensor([1, 2, 3, 4])
            keys = keys.view(1, -1).expand(X[:,1].long().size(0), -1)
            index = (X[:,1].long().unsqueeze(1) == keys).nonzero(as_tuple=True)
            # first learn, init memory states
            new_s = self.w[index[1]]
            new_d = self.w[4] - self.w[5] * (X[:,1] - 3)
            new_d = new_d.clamp(1, 10)
        else:
            r = power_forgetting_curve(X[:,0], state[:,0])
            new_d = state[:,1] - self.w[6] * (X[:,1] - 3)
            new_d = self.mean_reversion(self.w[4], new_d)
            new_d = new_d.clamp(1, 10)
            condition = X[:,1] > 1
            new_s = torch.where(condition, self.stability_after_success(state, new_d, r, X[:,1]), self.stability_after_failure(state, new_d, r))
        new_s = new_s.clamp(0.1, 36500)
        return torch.stack([new_s, new_d], dim=1)

    def forward(self, inputs: Tensor, state: Optional[Tensor]=None) -> Tensor:
        '''
        :param inputs: shape[seq_len, batch_size, 2]
        '''
        if state is None:
            state = torch.zeros((inputs.shape[1], 2))
        outputs = []
        for X in inputs:
            state = self.step(X, state)
            outputs.append(state)
        return torch.stack(outputs), state

    def mean_reversion(self, init: Tensor, current: Tensor) -> Tensor:
        return self.w[7] * init + (1-self.w[7]) * current

class WeightClipper:
    def __init__(self, frequency: int=1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, 'w'):
            w = module.w.data
            w[4] = w[4].clamp(1, 10)
            w[5] = w[5].clamp(0.1, 5)
            w[6] = w[6].clamp(0.1, 5)
            w[7] = w[7].clamp(0, 0.5)
            w[8] = w[8].clamp(0, 2)
            w[9] = w[9].clamp(0.1, 0.8)
            w[10] = w[10].clamp(0.01, 1.5)
            w[11] = w[11].clamp(0.5, 5)
            w[12] = w[12].clamp(0.01, 0.2)
            w[13] = w[13].clamp(0.01, 0.9)
            w[14] = w[14].clamp(0.01, 2)
            w[15] = w[15].clamp(0, 1)
            w[16] = w[16].clamp(1, 10)
            module.w.data = w

def lineToTensor(line: str) -> Tensor:
    ivl = line[0].split(',')
    response = line[1].split(',')
    tensor = torch.zeros(len(response), 2)
    for li, response in enumerate(response):
        tensor[li][0] = int(ivl[li])
        tensor[li][1] = int(response)
    return tensor

class RevlogDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        if dataframe.empty:
            raise ValueError('Training data is inadequate.')
        padded = pad_sequence(dataframe['tensor'].to_list(), batch_first=True, padding_value=0)
        self.x_train = padded.int()
        self.t_train = torch.tensor(dataframe['delta_t'].values, dtype=torch.int)
        self.y_train = torch.tensor(dataframe['y'].values, dtype=torch.float)
        self.seq_len = torch.tensor(dataframe['tensor'].map(len).values, dtype=torch.long)

    def __getitem__(self, idx):
        return self.x_train[idx], self.t_train[idx], self.y_train[idx], self.seq_len[idx]

    def __len__(self):
        return len(self.y_train)

class RevlogSampler(Sampler[List[int]]):
    def __init__(self, data_source: RevlogDataset, batch_size: int):
        self.data_source = data_source
        self.batch_size = batch_size
        lengths = np.array(data_source.seq_len)
        indices = np.argsort(lengths)
        full_batches, remainder = divmod(indices.size, self.batch_size)
        if full_batches > 0:
            if remainder == 0:
                self.batch_indices = np.split(indices, full_batches)
            else:
                self.batch_indices = np.split(indices[:-remainder], full_batches)
        else:
            self.batch_indices = []
        if remainder > 0:
            self.batch_indices.append(indices[-remainder:])
        self.batch_nums = len(self.batch_indices)
        # seed = int(torch.empty((), dtype=torch.int64).random_().item())
        seed = 2023
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __iter__(self):
        yield from (self.batch_indices[idx] for idx in torch.randperm(self.batch_nums, generator=self.generator).tolist())

    def __len__(self):
        return len(self.data_source)


def collate_fn(batch):
    sequences, delta_ts, labels, seq_lens = zip(*batch)
    sequences_packed = pack_padded_sequence(torch.stack(sequences, dim=1), lengths=torch.stack(seq_lens), batch_first=False, enforce_sorted=False)
    sequences_padded, length = pad_packed_sequence(sequences_packed, batch_first=False)
    sequences_padded = torch.as_tensor(sequences_padded)
    seq_lens = torch.as_tensor(length)
    delta_ts = torch.as_tensor(delta_ts)
    labels = torch.as_tensor(labels)
    return sequences_padded, delta_ts, labels, seq_lens

class Trainer:
    def __init__(self, train_set: pd.DataFrame, test_set: pd.DataFrame, init_w: List[float], n_epoch: int=1, lr: float=1e-2, batch_size: int=256) -> None:
        self.model = FSRS(init_w)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.clipper = WeightClipper()
        self.batch_size = batch_size
        self.build_dataset(train_set, test_set)
        self.n_epoch = n_epoch
        self.batch_nums = self.next_train_data_loader.batch_sampler.batch_nums
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.batch_nums * n_epoch)
        self.avg_train_losses = []
        self.avg_eval_losses = []
        self.loss_fn = nn.BCELoss(reduction='none')

    def build_dataset(self, train_set: pd.DataFrame, test_set: pd.DataFrame):
        pre_train_set = train_set[train_set['i'] == 2]
        self.pre_train_set = RevlogDataset(pre_train_set)
        sampler = RevlogSampler(self.pre_train_set, batch_size=self.batch_size)
        self.pre_train_data_loader = DataLoader(self.pre_train_set, batch_sampler=sampler, collate_fn=collate_fn)

        next_train_set = train_set[train_set['i'] > 2]
        self.next_train_set = RevlogDataset(next_train_set)
        sampler = RevlogSampler(self.next_train_set, batch_size=self.batch_size)
        self.next_train_data_loader = DataLoader(self.next_train_set, batch_sampler=sampler, collate_fn=collate_fn)

        self.train_set = RevlogDataset(train_set)
        sampler = RevlogSampler(self.train_set, batch_size=self.batch_size)
        self.train_data_loader = DataLoader(self.train_set, batch_sampler=sampler, collate_fn=collate_fn)

        self.test_set = RevlogDataset(test_set)
        sampler = RevlogSampler(self.test_set, batch_size=self.batch_size)
        self.test_data_loader = DataLoader(self.test_set, batch_sampler=sampler, collate_fn=collate_fn)
        tqdm.write("dataset built")

    def train(self, verbose: bool=True):
        best_loss = np.inf
        epoch_len = len(self.next_train_data_loader)
        pbar = tqdm(desc="train", colour="red", total=epoch_len*self.n_epoch)
        print_len = max(self.batch_nums*self.n_epoch // 10, 1)
        for k in range(self.n_epoch):
            weighted_loss, w = self.eval()
            if weighted_loss < best_loss:
                best_loss = weighted_loss
                best_w = w

            for i, batch in enumerate(self.next_train_data_loader):
                self.model.train()
                self.optimizer.zero_grad()
                sequences, delta_ts, labels, seq_lens = batch
                real_batch_size = seq_lens.shape[0]
                outputs, _ = self.model(sequences)
                stabilities = outputs[seq_lens-1, torch.arange(real_batch_size), 0]
                retentions = power_forgetting_curve(delta_ts, stabilities)
                loss = self.loss_fn(retentions, labels).sum()
                loss.backward()
                for param in self.model.parameters():
                    param.grad[:4] = torch.zeros(4)
                self.optimizer.step()
                self.scheduler.step()
                self.model.apply(self.clipper)
                pbar.update(real_batch_size)

                if verbose and (k * self.batch_nums + i + 1) % print_len == 0:
                    tqdm.write(f"iteration: {k * epoch_len + (i + 1) * self.batch_size}")
                    for name, param in self.model.named_parameters():
                        tqdm.write(f"{name}: {list(map(lambda x: round(float(x), 4),param))}")
        pbar.close()

        weighted_loss, w = self.eval()
        if weighted_loss < best_loss:
            best_loss = weighted_loss
            best_w = w

        return best_w

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            sequences, delta_ts, labels, seq_lens = self.train_set.x_train, self.train_set.t_train, self.train_set.y_train, self.train_set.seq_len
            real_batch_size = seq_lens.shape[0]
            outputs, _ = self.model(sequences.transpose(0, 1))
            stabilities = outputs[seq_lens-1, torch.arange(real_batch_size), 0]
            retentions = power_forgetting_curve(delta_ts, stabilities)
            tran_loss = self.loss_fn(retentions, labels).mean()
            self.avg_train_losses.append(tran_loss)

            sequences, delta_ts, labels, seq_lens = self.test_set.x_train, self.test_set.t_train, self.test_set.y_train, self.test_set.seq_len
            real_batch_size = seq_lens.shape[0]
            outputs, _ = self.model(sequences.transpose(0, 1))
            stabilities = outputs[seq_lens-1, torch.arange(real_batch_size), 0]
            retentions = power_forgetting_curve(delta_ts, stabilities)
            test_loss = self.loss_fn(retentions, labels).mean()
            self.avg_eval_losses.append(test_loss)

            w = list(map(lambda x: round(float(x), 4), dict(self.model.named_parameters())['w'].data))

            weighted_loss = (tran_loss * len(self.train_set) + test_loss * len(self.test_set)) / (len(self.train_set) + len(self.test_set))

            return weighted_loss, w

    def plot(self):
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(self.avg_train_losses, label='train')
        ax.plot(self.avg_eval_losses, label='test')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend()
        return fig

class Collection:
    def __init__(self, w: List[float]) -> None:
        self.model = FSRS(w)
        self.model.eval()

    def predict(self, t_history: str, r_history: str):
        with torch.no_grad():
            line_tensor = lineToTensor(list(zip([t_history], [r_history]))[0]).unsqueeze(1)
            output_t = self.model(line_tensor)
            return output_t[-1][0]

    def batch_predict(self, dataset):
        fast_dataset = RevlogDataset(dataset)
        with torch.no_grad():
            outputs, _ = self.model(fast_dataset.x_train.transpose(0, 1))
            stabilities, difficulties = outputs[fast_dataset.seq_len-1, torch.arange(len(fast_dataset))].transpose(0, 1)
            return stabilities.tolist(), difficulties.tolist()

"""Used to store all the results from FSRS related functions"""
class Optimizer:
    def __init__(self) -> None:
        tqdm.pandas()

    @staticmethod
    def anki_extract(filename: str):
        """Step 1"""
        # Extract the collection file or deck file to get the .anki21 database.
        with zipfile.ZipFile(f'{filename}', 'r') as zip_ref:
            zip_ref.extractall('./')
            tqdm.write("Deck file extracted successfully!")

    def create_time_series(self, timezone: str, revlog_start_date: str, next_day_starts_at: int, filter_out_suspended_cards: bool = False):
        """Step 2"""
        if os.path.isfile("collection.anki21b"):
            os.remove("collection.anki21b")
            raise Exception(
                "Please export the file with `support older Anki versions` if you use the latest version of Anki.")
        elif os.path.isfile("collection.anki21"):
            con = sqlite3.connect("collection.anki21")
        elif os.path.isfile("collection.anki2"):
            con = sqlite3.connect("collection.anki2")
        else:
            raise Exception("Collection not exist!")
        cur = con.cursor()
        res = cur.execute(f"""
        SELECT *
        FROM revlog
        WHERE cid IN (
            SELECT id
            FROM cards
            WHERE queue != 0
            {"AND queue != -1" if filter_out_suspended_cards else ""}
        )
        """
        )
        revlog = res.fetchall()
        if len(revlog) == 0:
            raise Exception("No review log found!")
        df = pd.DataFrame(revlog)
        df.columns = ['id', 'cid', 'usn', 'r', 'ivl', 'last_ivl', 'factor', 'time', 'type']
        df = df[(df['cid'] <= time.time() * 1000) &
                (df['id'] <= time.time() * 1000)].copy()
        
        df_set_due_date = df[(df['type'] == 4) & (df['ivl'] > 0)]
        df.drop(df_set_due_date.index, inplace=True)

        df['create_date'] = pd.to_datetime(df['cid'] // 1000, unit='s')
        df['create_date'] = df['create_date'].dt.tz_localize('UTC').dt.tz_convert(timezone)
        df['review_date'] = pd.to_datetime(df['id'] // 1000, unit='s')
        df['review_date'] = df['review_date'].dt.tz_localize('UTC').dt.tz_convert(timezone)
        df.drop(df[df['review_date'].dt.year < 2006].index, inplace=True)
        df.sort_values(by=['cid', 'id'], inplace=True, ignore_index=True)

        df['is_learn_start'] = (df['type'] == 0) & (df['type'].shift() != 0)
        df['sequence_group'] = df['is_learn_start'].cumsum()
        last_learn_start = df[df['is_learn_start']].groupby('cid')['sequence_group'].last()
        df['last_learn_start'] = df['cid'].map(last_learn_start).fillna(0).astype(int)
        df['mask'] = df['last_learn_start'] <= df['sequence_group']
        df = df[df['mask'] == True].copy()
        df.drop(columns=['is_learn_start', 'sequence_group', 'last_learn_start', 'mask'], inplace=True)
        df = df[(df['type'] != 4)].copy()

        self.type_sequence = np.array(df['type'])
        self.time_sequence = np.array(df['time'])
        df.to_csv("revlog.csv", index=False)
        tqdm.write("revlog.csv saved.")

        df = df[(df['type'] != 3) | (df['factor'] != 0)].copy()
        df['real_days'] = df['review_date'] - timedelta(hours=int(next_day_starts_at))
        df['real_days'] = pd.DatetimeIndex(df['real_days'].dt.floor('D', ambiguous='infer', nonexistent='shift_forward')).to_julian_date()
        df.drop_duplicates(['cid', 'real_days'], keep='first', inplace=True)
        df['delta_t'] = df.real_days.diff()
        df.dropna(inplace=True)
        df['i'] = df.groupby('cid').cumcount() + 1
        df.loc[df['i'] == 1, 'delta_t'] = 0
        df = df.groupby('cid').filter(lambda group: group['type'].iloc[0] == 0)
        df['prev_type'] = df.groupby('cid')['type'].shift(1).fillna(0).astype(int)
        df['helper'] = ((df['type'] == 0) & ((df['prev_type'] == 1) | (df['prev_type'] == 2)) & (df['i'] > 1)).astype(int)
        df['helper'] = df.groupby('cid')['helper'].cumsum()
        df = df[df['helper'] == 0]
        del df['prev_type']
        del df['helper']

        def cum_concat(x):
            return list(accumulate(x))

        t_history = df.groupby('cid', group_keys=False)['delta_t'].apply(lambda x: cum_concat([[int(i)] for i in x]))
        df['t_history']=[','.join(map(str, item[:-1])) for sublist in t_history for item in sublist]
        r_history = df.groupby('cid', group_keys=False)['r'].apply(lambda x: cum_concat([[i] for i in x]))
        df['r_history']=[','.join(map(str, item[:-1])) for sublist in r_history for item in sublist]
        df = df.groupby('cid').filter(lambda group: group['id'].min() > time.mktime(datetime.strptime(revlog_start_date, "%Y-%m-%d").timetuple()) * 1000)
        df['y'] = df['r'].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])
        df.to_csv('revlog_history.tsv', sep="\t", index=False)
        tqdm.write("Trainset saved.")

        S0_dataset = df[df['i'] == 2]
        self.S0_dataset_group = S0_dataset.groupby(by=['r_history', 'delta_t'], group_keys=False).agg({'y': ['mean', 'count']}).reset_index()

        df['retention'] = df.groupby(by=['r_history', 'delta_t'], group_keys=False)['y'].transform('mean')
        df['total_cnt'] = df.groupby(by=['r_history', 'delta_t'], group_keys=False)['id'].transform('count')
        tqdm.write("Retention calculated.")

        df = df.drop(columns=['id', 'cid', 'usn', 'ivl', 'last_ivl', 'factor', 'time', 'type', 'create_date', 'review_date', 'real_days', 'r', 't_history', 'y'])
        df.drop_duplicates(inplace=True)
        df['retention'] = df['retention'].map(lambda x: max(min(0.99, x), 0.01))

        def cal_stability(group: pd.DataFrame) -> pd.DataFrame:
            group_cnt = sum(group['total_cnt'])
            if group_cnt < 10:
                return pd.DataFrame()
            group['group_cnt'] = group_cnt
            if group['i'].values[0] > 1:
                r_ivl_cnt = sum(group['delta_t'] * group['retention'].map(np.log) * pow(group['total_cnt'], 2))
                ivl_ivl_cnt = sum(group['delta_t'].map(lambda x: x ** 2) * pow(group['total_cnt'], 2))
                group['stability'] = round(np.log(0.9) / (r_ivl_cnt / ivl_ivl_cnt), 1)
            else:
                group['stability'] = 0.0
            group['avg_retention'] = round(sum(group['retention'] * pow(group['total_cnt'], 2)) / sum(pow(group['total_cnt'], 2)), 3)
            group['avg_interval'] = round(sum(group['delta_t'] * pow(group['total_cnt'], 2)) / sum(pow(group['total_cnt'], 2)), 1)
            del group['total_cnt']
            del group['retention']
            del group['delta_t']
            return group

        df = df.groupby(by=['r_history'], group_keys=False).progress_apply(cal_stability)
        tqdm.write("Stability calculated.")
        df.reset_index(drop = True, inplace = True)
        df.drop_duplicates(inplace=True)
        df.sort_values(by=['r_history'], inplace=True, ignore_index=True)

        if df.shape[0] > 0:
            for idx in tqdm(df.index, desc="analysis"):
                item = df.loc[idx]
                index = df[(df['i'] == item['i'] + 1) & (df['r_history'].str.startswith(item['r_history']))].index
                df.loc[index, 'last_stability'] = item['stability']
            df['factor'] = round(df['stability'] / df['last_stability'], 2)
            df = df[(df['i'] >= 2) & (df['group_cnt'] >= 100)].copy()
            df['last_recall'] = df['r_history'].map(lambda x: x[-1])
            df = df[df.groupby(['i', 'r_history'], group_keys=False)['group_cnt'].transform(max) == df['group_cnt']]
            df.to_csv('./stability_for_analysis.tsv', sep='\t', index=None)
            tqdm.write("Analysis saved!")
            caption = "1:again, 2:hard, 3:good, 4:easy\n"
            analysis = df[df['r_history'].str.contains(r'^[1-4][^124]*$', regex=True)][['r_history', 'avg_interval', 'avg_retention', 'stability', 'factor', 'group_cnt']].to_string(index=False)
            return caption + analysis

    def define_model(self):
        """Step 3"""
        self.init_w = [1, 2, 3, 4, 5, 0.5, 0.5, 0.2, 1.4, 0.2, 0.8, 2, 0.2, 0.2, 1, 0.5, 2]
        '''
        w[0]: initial_stability_for_again_answer
        w[1]: initial_stability_step_per_rating
        w[2]: initial_difficulty_for_good_answer
        w[3]: initial_difficulty_step_per_rating
        w[4]: next_difficulty_step_per_rating
        w[5]: next_difficulty_reversion_to_mean_speed (used to avoid ease hell)
        w[6]: next_stability_factor_after_success
        w[7]: next_stability_stabilization_decay_after_success
        w[8]: next_stability_retrievability_gain_after_success
        w[9]: next_stability_factor_after_failure
        w[10]: next_stability_difficulty_decay_after_success
        w[11]: next_stability_stability_gain_after_failure
        w[12]: next_stability_retrievability_gain_after_failure
        For more details about the parameters, please see: 
        https://github.com/open-spaced-repetition/fsrs4anki/wiki/Free-Spaced-Repetition-Scheduler
        '''

    def pretrain(self, verbose=True):
        rating_stability = {}
        rating_count = {}

        for first_rating in ("1", "2", "3", "4"):
            group = self.S0_dataset_group[self.S0_dataset_group['r_history'] == first_rating]
            if group.empty:
                continue
            delta_t = group['delta_t']
            recall = group['y']['mean']
            count = group['y']['count']
            total_count = sum(count)
            if total_count < 100:
                tqdm.write(f'Not enough data for first rating {first_rating}. Expected at least 100, got {total_count}.')
                continue
            params, _ = curve_fit(power_forgetting_curve, delta_t, recall, sigma=1/np.sqrt(count), bounds=((0.1), (365)))
            stability = params[0]
            rating_stability[int(first_rating)] = stability
            rating_count[int(first_rating)] = total_count
            predict_recall = power_forgetting_curve(delta_t, *params)
            rmse = mean_squared_error(recall, predict_recall, sample_weight=count, squared=False)

            if verbose:
                plt.plot(delta_t, recall, label='Exact')
                plt.plot(np.linspace(0, 30), power_forgetting_curve(np.linspace(0, 30), *params), label=f'Weighted fit (RMSE: {rmse:.4f})')
                count_percent = np.array([x/total_count for x in count])
                plt.scatter(delta_t, recall, s=count_percent * 1000, alpha=0.5)
                plt.legend(loc='upper right', fancybox=True, shadow=False)
                plt.grid(True)
                plt.ylim(0, 1)
                plt.xlim(0, 30)
                plt.xlabel('Interval')
                plt.ylabel('Recall')
                plt.title(f'Forgetting curve for first rating {first_rating} (n={total_count}, s={stability:.2f})')
                plt.show()
                tqdm.write(str(rating_stability))

        if len(rating_stability) < 2:
            raise Exception("Not enough data for pretraining!")

        def S0_rating_curve(rating, a, b, c):
            return np.exp(a + b * rating) + c

        params, covs = curve_fit(S0_rating_curve, list(rating_stability.keys()), list(rating_stability.values()), sigma=1/np.sqrt(list(rating_count.values())), method='dogbox', bounds=((-15, 0.03, -5), (15, 7, 30)))
        if verbose:
            tqdm.write(f'Weighted fit parameters: {params}')
            predict_stability = S0_rating_curve(np.array(list(rating_stability.keys())), *params)
            tqdm.write(f"Fit stability: {predict_stability}")
            tqdm.write(f'RMSE: {mean_squared_error(list(rating_stability.values()), predict_stability, sample_weight=list(rating_count.values()), squared=False):.4f}')
            plt.plot(list(rating_stability.keys()), list(rating_stability.values()), label='Exact')
            plt.plot(list(rating_stability.keys()), predict_stability, label='Weighted fit')
            plt.legend(loc='upper right', fancybox=True, shadow=False)
            plt.grid(True)
            plt.xlabel('First rating')
            plt.ylabel('Stability')
            plt.title('Stability for first rating')
            plt.show()

        for rating in (1, 2, 3, 4):
            again_extrap = max(min(S0_rating_curve(1, *params), 365), 0.1)
            # if there isn't enough data to calculate the value for "Again" exactly
            if 1 not in rating_stability:
                # then check if there exists an exact value for "Hard"
                if 2 in rating_stability:
                    # if it exists, then check whether the extrapolation breaks monotonicity
                    # Again > Hard is possible, but we should allow it only for exact values, otherwise we should assume monotonicity
                    if again_extrap > rating_stability[2]:
                        # if it does, then replace the missing "Again" value with the exact "Hard" value
                        rating_stability[1] = rating_stability[2]
                    else:
                        # if it doesn't break monotonicity, then use the extrapolated value
                        rating_stability[1] = again_extrap
                # if an exact value for "Hard" doesn't exist, then just use the extrapolation, there's nothing else we can do
                else:
                    rating_stability[1] = again_extrap
            elif rating not in rating_stability:
                rating_stability[rating] = max(min(S0_rating_curve(rating, *params), 365), 0.1)

        rating_stability = {k: round(v, 2) for k, v in sorted(rating_stability.items(), key=lambda item: item[0])}
        for rating, stability in rating_stability.items():
            self.init_w[rating-1] = stability
        tqdm.write(f"Pretrain finished!")

    def train(self, lr: float = 4e-2, n_epoch: int = 5, n_splits: int = 5, batch_size: int = 512, verbose: bool = True):
        """Step 4"""
        self.dataset = pd.read_csv("./revlog_history.tsv", sep='\t', index_col=None, dtype={'r_history': str ,'t_history': str} )
        self.dataset = self.dataset[(self.dataset['i'] > 1) & (self.dataset['delta_t'] > 0) & (self.dataset['t_history'].str.count(',0') == 0)]
        if self.dataset.empty:
            raise ValueError('Training data is inadequate.')
        self.dataset['tensor'] = self.dataset.progress_apply(lambda x: lineToTensor(list(zip([x['t_history']], [x['r_history']]))[0]), axis=1)
        self.dataset['group'] = self.dataset['r_history'] + self.dataset['t_history']
        tqdm.write("Tensorized!")

        w = []
        plots = []
        if n_splits > 1:
            sgkf = StratifiedGroupKFold(n_splits=n_splits)
            for train_index, test_index in sgkf.split(self.dataset, self.dataset['i'], self.dataset['group']):
                tqdm.write(f"TRAIN: {len(train_index)} TEST: {len(test_index)}")
                train_set = self.dataset.iloc[train_index].copy()
                test_set = self.dataset.iloc[test_index].copy()
                trainer = Trainer(train_set, test_set, self.init_w, n_epoch=n_epoch, lr=lr, batch_size=batch_size)
                w.append(trainer.train(verbose=verbose))
                plots.append(trainer.plot())
        else:
            trainer = Trainer(self.dataset, self.dataset, self.init_w, n_epoch=n_epoch, lr=lr, batch_size=batch_size)
            w.append(trainer.train(verbose=verbose))
            plots.append(trainer.plot())

        w = np.array(w)
        avg_w = np.round(np.mean(w, axis=0), 4)
        self.w = avg_w.tolist()

        tqdm.write("\nTraining finished!")
        return plots

    def preview(self, requestRetention: float):
        my_collection = Collection(self.w)
        preview_text = "1:again, 2:hard, 3:good, 4:easy\n"
        for first_rating in (1,2,3,4):
            preview_text += f'\nfirst rating: {first_rating}\n'
            t_history = "0"
            d_history = "0"
            r_history = f"{first_rating}"  # the first rating of the new card
            # print("stability, difficulty, lapses")
            for i in range(10):
                states = my_collection.predict(t_history, r_history)
                # print('{0:9.2f} {1:11.2f} {2:7.0f}'.format(
                    # *list(map(lambda x: round(float(x), 4), states))))
                next_t = next_interval(states[0], requestRetention)                
                difficulty = round(float(states[1]), 1)
                t_history += f',{int(next_t)}'
                d_history += f',{difficulty}'
                r_history += f",3"
            preview_text += f"rating history: {r_history}\n"
            preview_text += "interval history: " + ",".join([f"{ivl}d" if ivl < 30 else f"{ivl / 30:.1f}m" if ivl < 365 else f"{ivl / 365:.1f}y" for ivl in map(int, t_history.split(','))]) + "\n"
            preview_text += f"difficulty history: {d_history}\n"
        return preview_text

    def preview_sequence(self, test_rating_sequence: str, requestRetention: float):
        my_collection = Collection(self.w)

        t_history = "0"
        d_history = "0"
        for i in range(len(test_rating_sequence.split(','))):
            r_history = test_rating_sequence[:2*i+1]
            states = my_collection.predict(t_history, r_history)
            next_t = next_interval(states[0], requestRetention)
            t_history += f',{int(next_t)}'
            difficulty = round(float(states[1]), 1)
            d_history += f',{difficulty}'
        preview_text = f"rating history: {test_rating_sequence}\n"
        preview_text += f"interval history: {t_history}\n"
        preview_text += f"difficulty history: {d_history}"
        return preview_text

    def predict_memory_states(self):
        my_collection = Collection(self.w)

        stabilities, difficulties = my_collection.batch_predict(self.dataset)
        stabilities = map(lambda x: round(x, 2), stabilities)
        difficulties = map(lambda x: round(x, 2), difficulties)
        self.dataset['stability'] = list(stabilities)
        self.dataset['difficulty'] = list(difficulties)
        prediction = self.dataset.groupby(by=['t_history', 'r_history']).agg({"stability": "mean", "difficulty": "mean", "id": "count"})
        prediction.reset_index(inplace=True)
        prediction.sort_values(by=['r_history'], inplace=True)
        prediction.rename(columns={"id": "count"}, inplace=True)
        prediction.to_csv("./prediction.tsv", sep='\t', index=None)
        tqdm.write("prediction.tsv saved.")
        prediction['difficulty'] = prediction['difficulty'].map(lambda x: int(round(x)))
        self.difficulty_distribution = prediction.groupby(by=['difficulty'])['count'].sum() / prediction['count'].sum()
        self.difficulty_distribution_padding = np.zeros(10)
        for i in range(10):
            if i+1 in self.difficulty_distribution.index:
                self.difficulty_distribution_padding[i] = self.difficulty_distribution.loc[i+1]
        return self.difficulty_distribution
    
    def find_optimal_retention(self):
        """should not be called before predict_memory_states"""

        base = 1.01
        minimum_stability = 0.1
        index_offset = -(np.log(minimum_stability) / np.log(base)).round().astype(int)
        d_range = 10
        d_offset = 1
        r_time = 8
        f_time = 25
        max_time = 1e10

        type_block = dict()
        type_count = dict()
        type_time = dict()
        last_t = self.type_sequence[0]
        type_block[last_t] = 1
        type_count[last_t] = 1
        type_time[last_t] = self.time_sequence[0]
        for i,t in enumerate(self.type_sequence[1:]):
            type_count[t] = type_count.setdefault(t, 0) + 1
            type_time[t] = type_time.setdefault(t, 0) + self.time_sequence[i]
            if t != last_t:
                type_block[t] = type_block.setdefault(t, 0) + 1
            last_t = t

        r_time = round(type_time[1]/type_count[1]/1000, 1)

        if 2 in type_count and 2 in type_block:
            f_time = round(type_time[2]/type_block[2]/1000 + r_time, 1)

        tqdm.write(f"average time for failed cards: {f_time}s")
        tqdm.write(f"average time for recalled cards: {r_time}s")

        def stability2index(stability):
            return (np.log(stability) / np.log(base)).round().astype(int) + index_offset

        def cal_next_recall_stability(s, r, d, response):
            if response == 1:
                return s * (1 + np.exp(self.w[8]) * (11 - d) * np.power(s, -self.w[9]) * (np.exp((1 - r) * self.w[10]) - 1))
            else:
                return np.minimum(self.w[11] * np.power(d, -self.w[12]) * (np.power(s + 1, self.w[13]) - 1) * np.exp((1 - r) * self.w[14]), s)

        terminal_stability = minimum_stability
        for _ in range(128):
            terminal_stability = cal_next_recall_stability(terminal_stability, 0.96, d_range, 1)
        index_len = stability2index(terminal_stability)
        stability_list = np.array([np.power(base, i - index_offset) for i in range(index_len)])
        tqdm.write(f"terminal stability: {stability_list.max(): .2f}")
        df = pd.DataFrame(columns=["retention", "difficulty", "time"])

        for percentage in tqdm(range(96, 66, -2), desc="find optimal retention"):
            recall = percentage / 100
            time_list = np.zeros((d_range, index_len))
            time_list[:,:-1] = max_time
            for d in range(d_range, 0, -1):
                s0 = 1
                s0_index = stability2index(s0)
                diff = max_time
                iteration = 0
                while diff > 1 and iteration < 2e5:
                    iteration += 1
                    total_time = time_list[d - 1].sum()
                    s_indices = np.arange(index_len - 2, -1, -1)
                    stabilities = stability_list[s_indices]
                    intervals = next_interval(stabilities, recall)
                    p_recalls = power_forgetting_curve(intervals, stabilities)
                    recall_s = cal_next_recall_stability(stabilities, p_recalls, d, 1)
                    forget_d = np.minimum(d + d_offset, 10)
                    forget_s = cal_next_recall_stability(stabilities, p_recalls, forget_d, 0)
                    recall_s_indices = np.minimum(stability2index(recall_s), index_len - 1)
                    forget_s_indices = np.clip(stability2index(forget_s), 0, index_len - 1)
                    recall_times = time_list[d - 1][recall_s_indices] + r_time
                    forget_times = time_list[forget_d - 1][forget_s_indices] + f_time
                    exp_times = p_recalls * recall_times + (1.0 - p_recalls) * forget_times
                    mask = exp_times < time_list[d - 1][s_indices]
                    time_list[d - 1][s_indices[mask]] = exp_times[mask]
                    diff = total_time - time_list[d - 1].sum()
                    s0_time = time_list[d - 1][s0_index]
                df.loc[0 if pd.isnull(df.index.max()) else df.index.max() + 1] = [recall, d, s0_time]

        df.sort_values(by=["difficulty", "retention"], inplace=True)
        df.to_csv("./expected_time.csv", index=False)
        tqdm.write("expected_time.csv saved.")

        optimal_retention_list = np.zeros(10)
        fig = plt.figure()
        ax = fig.gca()
        for d in range(1, d_range+1):
            retention = df[df["difficulty"] == d]["retention"]
            cost = df[df["difficulty"] == d]["time"]
            optimal_retention = retention.iat[cost.argmin()]
            optimal_retention_list[d-1] = optimal_retention
            ax.plot(retention, cost, label=f"d={d}, r={optimal_retention}")
        
        self.optimal_retention = np.inner(self.difficulty_distribution_padding, optimal_retention_list)

        tqdm.write(f"\n-----suggested retention (experimental): {self.optimal_retention:.2f}-----")

        ax.set_ylabel("expected time (second)")
        ax.set_xlabel("retention")
        ax.legend()
        ax.grid()
        ax.semilogy()
        return (fig, )
    
    def evaluate(self):
        my_collection = Collection(self.init_w)
        stabilities, difficulties = my_collection.batch_predict(self.dataset)
        self.dataset['stability'] = stabilities
        self.dataset['difficulty'] = difficulties
        self.dataset['p'] = power_forgetting_curve(self.dataset['delta_t'], self.dataset['stability'])
        self.dataset['log_loss'] = self.dataset.apply(lambda row: - np.log(row['p']) if row['y'] == 1 else - np.log(1 - row['p']), axis=1)
        loss_before = self.dataset['log_loss'].mean()

        my_collection = Collection(self.w)
        stabilities, difficulties = my_collection.batch_predict(self.dataset)
        self.dataset['stability'] = stabilities
        self.dataset['difficulty'] = difficulties
        self.dataset['p'] = power_forgetting_curve(self.dataset['delta_t'], self.dataset['stability'])
        self.dataset['log_loss'] = self.dataset.apply(lambda row: - np.log(row['p']) if row['y'] == 1 else - np.log(1 - row['p']), axis=1)
        loss_after = self.dataset['log_loss'].mean()

        tmp = self.dataset.copy()
        tmp['stability'] = tmp['stability'].map(lambda x: round(x, 2))
        tmp['difficulty'] = tmp['difficulty'].map(lambda x: round(x, 2))
        tmp['p'] = tmp['p'].map(lambda x: round(x, 2))
        tmp['log_loss'] = tmp['log_loss'].map(lambda x: round(x, 2))
        tmp.rename(columns={"r": "grade", "p": "retrievability"}, inplace=True)
        tmp[['id', 'cid', 'review_date', 'r_history', 't_history', 'delta_t', 'grade', 'stability', 'difficulty', 'retrievability', 'log_loss']].to_csv("./evaluation.tsv", sep='\t', index=False)
        del tmp
        return loss_before, loss_after

    def calibration_graph(self):
        fig1 = plt.figure()
        plot_brier(self.dataset['p'], self.dataset['y'], bins=40, ax=fig1.add_subplot(111))
        fig2 = plt.figure(figsize=(16, 12))
        for last_rating in ("1","2","3","4"):
            tqdm.write(f"\nLast rating: {last_rating}")
            plot_brier(self.dataset[self.dataset['r_history'].str.endswith(last_rating)]['p'], self.dataset[self.dataset['r_history'].str.endswith(last_rating)]['y'], bins=40, ax=fig2.add_subplot(2, 2, int(last_rating)), title=f"Last rating: {last_rating}")

        def to_percent(temp, position):
            return '%1.0f' % (100 * temp) + '%'

        fig3 = plt.figure()
        ax1 = fig3.add_subplot(111)
        ax2 = ax1.twinx()
        lns = []

        stability_calibration = pd.DataFrame(columns=['stability', 'predicted_retention', 'actual_retention'])
        stability_calibration = self.dataset[['stability', 'p', 'y']].copy()
        stability_calibration['bin'] = stability_calibration['stability'].map(lambda x: math.pow(1.2, math.floor(math.log(x, 1.2))))
        stability_group = stability_calibration.groupby('bin').count()

        lns1 = ax1.bar(x=stability_group.index, height=stability_group['y'], width=stability_group.index / 5.5,
                        ec='k', lw=.2, label='Number of predictions', alpha=0.5)
        ax1.set_ylabel("Number of predictions")
        ax1.set_xlabel("Stability (days)")
        ax1.semilogx()
        lns.append(lns1)

        stability_group = stability_calibration.groupby(by='bin').agg('mean')
        lns2 = ax2.plot(stability_group['y'], label='Actual retention')
        lns3 = ax2.plot(stability_group['p'], label='Predicted retention')
        ax2.set_ylabel("Retention")
        ax2.set_ylim(0, 1)
        lns.append(lns2[0])
        lns.append(lns3[0])

        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc='lower right')
        ax2.grid(linestyle='--')
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
        ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

        fig4 = plt.figure()
        ax1 = fig4.add_subplot(111)
        ax2 = ax1.twinx()
        lns = []

        difficulty_calibration = pd.DataFrame(columns=['difficulty', 'predicted_retention', 'actual_retention'])
        difficulty_calibration = self.dataset[['difficulty', 'p', 'y']].copy()
        difficulty_calibration['bin'] = difficulty_calibration['difficulty'].map(round)
        difficulty_group = difficulty_calibration.groupby('bin').count()

        lns1 = ax1.bar(x=difficulty_group.index, height=difficulty_group['y'],
                        ec='k', lw=.2, label='Number of predictions', alpha=0.5)
        ax1.set_ylabel("Number of predictions")
        ax1.set_xlabel("Difficulty")
        lns.append(lns1)

        difficulty_group = difficulty_calibration.groupby(by='bin').agg('mean')
        lns2 = ax2.plot(difficulty_group['y'], label='Actual retention')
        lns3 = ax2.plot(difficulty_group['p'], label='Predicted retention')
        ax2.set_ylabel("Retention")
        ax2.set_ylim(0, 1)
        lns.append(lns2[0])
        lns.append(lns3[0])

        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc='lower right')
        ax2.grid(linestyle='--')
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
        ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

        return fig1, fig2, fig3, fig4

    def bw_matrix(self):
        B_W_Metric_raw = self.dataset[['difficulty', 'stability', 'p', 'y']].copy()
        B_W_Metric_raw['s_bin'] = B_W_Metric_raw['stability'].map(lambda x: round(math.pow(1.4, math.floor(math.log(x, 1.4))), 2))
        B_W_Metric_raw['d_bin'] = B_W_Metric_raw['difficulty'].map(lambda x: int(round(x)))
        B_W_Metric = B_W_Metric_raw.groupby(by=['s_bin', 'd_bin']).agg('mean').reset_index()
        B_W_Metric_count = B_W_Metric_raw.groupby(by=['s_bin', 'd_bin']).agg('count').reset_index()
        B_W_Metric['B-W'] = B_W_Metric['p'] - B_W_Metric['y']
        n = len(self.dataset)
        bins = len(B_W_Metric)
        B_W_Metric_pivot = B_W_Metric[B_W_Metric_count['p'] > max(50, n / (3 * bins))].pivot(index="s_bin", columns='d_bin', values='B-W')
        return B_W_Metric_pivot.apply(pd.to_numeric).style.background_gradient(cmap='seismic', axis=None, vmin=-0.2, vmax=0.2).format("{:.2%}", na_rep='')

    def compare_with_sm2(self):
        self.dataset['sm2_ivl'] = self.dataset['tensor'].map(sm2)
        self.dataset['sm2_p'] = np.exp(np.log(0.9) * self.dataset['delta_t'] / self.dataset['sm2_ivl'])
        self.dataset['log_loss'] = self.dataset.apply(lambda row: - np.log(row['sm2_p']) if row['y'] == 1 else - np.log(1 - row['sm2_p']), axis=1)
        tqdm.write(f"Loss of SM-2: {self.dataset['log_loss'].mean():.4f}")
        cross_comparison = self.dataset[['sm2_p', 'p', 'y']].copy()
        fig1 = plt.figure()
        plot_brier(cross_comparison['sm2_p'], cross_comparison['y'], bins=40, ax=fig1.add_subplot(111))

        fig2 = plt.figure(figsize=(6, 6))
        ax = fig2.gca()

        cross_comparison['SM2_B-W'] = cross_comparison['sm2_p'] - cross_comparison['y']
        cross_comparison['SM2_bin'] = cross_comparison['sm2_p'].map(lambda x: round(x, 1))
        cross_comparison['FSRS_B-W'] = cross_comparison['p'] - cross_comparison['y']
        cross_comparison['FSRS_bin'] = cross_comparison['p'].map(lambda x: round(x, 1))

        ax.axhline(y = 0.0, color = 'black', linestyle = '-')

        cross_comparison_group = cross_comparison.groupby(by='SM2_bin').agg({'y': ['mean'], 'FSRS_B-W': ['mean'], 'p': ['mean', 'count']})
        tqdm.write(f"Universal Metric of FSRS: {mean_squared_error(cross_comparison_group['y', 'mean'], cross_comparison_group['p', 'mean'], sample_weight=cross_comparison_group['p', 'count'], squared=False):.4f}")
        cross_comparison_group['p', 'percent'] = cross_comparison_group['p', 'count'] / cross_comparison_group['p', 'count'].sum()
        ax.scatter(cross_comparison_group.index, cross_comparison_group['FSRS_B-W', 'mean'], s=cross_comparison_group['p', 'percent'] * 1024, alpha=0.5)
        ax.plot(cross_comparison_group['FSRS_B-W', 'mean'], label='FSRS by SM2')

        cross_comparison_group = cross_comparison.groupby(by='FSRS_bin').agg({'y': ['mean'], 'SM2_B-W': ['mean'], 'sm2_p': ['mean', 'count']})
        tqdm.write(f"Universal Metric of SM2: {mean_squared_error(cross_comparison_group['y', 'mean'], cross_comparison_group['sm2_p', 'mean'], sample_weight=cross_comparison_group['sm2_p', 'count'], squared=False):.4f}")
        cross_comparison_group['sm2_p', 'percent'] = cross_comparison_group['sm2_p', 'count'] / cross_comparison_group['sm2_p', 'count'].sum()
        ax.scatter(cross_comparison_group.index, cross_comparison_group['SM2_B-W', 'mean'], s=cross_comparison_group['sm2_p', 'percent'] * 1024, alpha=0.5)
        ax.plot(cross_comparison_group['SM2_B-W', 'mean'], label='SM2 by FSRS')

        ax.legend(loc='lower center')
        ax.grid(linestyle='--')
        ax.set_title("SM2 vs. FSRS")
        ax.set_xlabel('Predicted R')
        ax.set_ylabel('B-W Metric')
        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        return fig1, fig2

# code from https://github.com/papousek/duolingo-halflife-regression/blob/master/evaluation.py
def load_brier(predictions, real, bins=20):
    counts = np.zeros(bins)
    correct = np.zeros(bins)
    prediction = np.zeros(bins)
    for p, r in zip(predictions, real):
        bin = min(int(p * bins), bins - 1)
        counts[bin] += 1
        correct[bin] += r
        prediction[bin] += p
    np.seterr(invalid='ignore')
    prediction_means = prediction / counts
    prediction_means[np.isnan(prediction_means)] = ((np.arange(bins) + 0.5) / bins)[np.isnan(prediction_means)]
    correct_means = correct / counts
    correct_means[np.isnan(correct_means)] = 0
    size = len(predictions)
    answer_mean = sum(correct) / size
    return {
        "reliability": sum(counts * (correct_means - prediction_means) ** 2) / size,
        "resolution": sum(counts * (correct_means - answer_mean) ** 2) / size,
        "uncertainty": answer_mean * (1 - answer_mean),
        "detail": {
            "bin_count": bins,
            "bin_counts": list(counts),
            "bin_prediction_means": list(prediction_means),
            "bin_correct_means": list(correct_means),
        }
    }

def plot_brier(predictions, real, bins=20, ax=None, title=None):
    brier = load_brier(predictions, real, bins=bins)
    bin_prediction_means = brier['detail']['bin_prediction_means']
    bin_correct_means = brier['detail']['bin_correct_means']
    bin_counts = brier['detail']['bin_counts']
    r2 = r2_score(bin_correct_means, bin_prediction_means, sample_weight=bin_counts)
    rmse = np.sqrt(mean_squared_error(bin_correct_means, bin_prediction_means, sample_weight=bin_counts))
    tqdm.write(f"R-squared: {r2:.4f}")
    tqdm.write(f"RMSE: {rmse:.4f}")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True)
    fit_wls = sm.WLS(bin_correct_means, sm.add_constant(bin_prediction_means), weights=bin_counts).fit()
    tqdm.write(str(fit_wls.params))
    y_regression = [fit_wls.params[0] + fit_wls.params[1]*x for x in bin_prediction_means]
    ax.plot(bin_prediction_means, y_regression, label='Weighted Least Squares Regression', color="green")
    ax.plot(bin_prediction_means, bin_correct_means, label='Actual Calibration', color="#1f77b4")
    ax.plot((0, 1), (0, 1), label='Perfect Calibration', color="#ff7f0e")
    bin_count = brier['detail']['bin_count']
    counts = np.array(bin_counts)
    bins = (np.arange(bin_count) + 0.5) / bin_count
    ax.legend(loc='upper center')
    ax.set_xlabel('Predicted R')
    ax.set_ylabel('Actual R')
    ax2 = ax.twinx()
    ax2.set_ylabel('Number of reviews')
    ax2.bar(bins, counts, width=(0.8 / bin_count), ec='k', lw=.2, alpha=0.5, label='Number of reviews')
    ax2.legend(loc='lower center')
    if title:
        ax.set_title(title)

def sm2(history):
    ivl = 0
    ef = 2.5
    reps = 0
    for delta_t, rating in history:
        delta_t = delta_t.item()
        rating = rating.item() + 1
        if rating > 2:
            if reps == 0:
                ivl = 1
                reps = 1
            elif reps == 1:
                ivl = 6
                reps = 2
            else:
                ivl = ivl * ef
                reps += 1
        else:
            ivl = 1
            reps = 0
        ef = max(1.3, ef + (0.1 - (5 - rating) * (0.08 + (5 - rating) * 0.02)))
        ivl = max(1, round(ivl+0.01))
    return ivl