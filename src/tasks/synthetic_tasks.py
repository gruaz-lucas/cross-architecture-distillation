import torch
from torch.utils.data import Dataset, DataLoader

class DelayedCopyTask(Dataset):
    """
    Task:
      Input:  [study_seq, delay_seq, prompt_tokens]
      Target: [ignore_tokens, ..., study_seq]
    """
    def __init__(self, vocab_size=10, seq_len=5, delay=3, n_samples=10000):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.delay = delay
        self.n_samples = n_samples
        self.pad_token = 0
        self.prompt_token = vocab_size + 1
        self.ignore_token = vocab_size + 2

        # Effective vocab = vocab tokens (1..vocab_size) + special tokens
        self.total_vocab_size = vocab_size + 3  # includes pad, prompt, ignore

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        study_seq = torch.randint(1, self.vocab_size + 1, (self.seq_len,))

        # Build input: [study_seq, delay_tokens, prompt_tokens]
        delay_seq = torch.full((self.delay,), self.pad_token)
        prompt_seq = torch.full((self.seq_len,), self.prompt_token)
        x = torch.cat([study_seq, delay_seq, prompt_seq], dim=0)

        # Build target: [ignore_tokens (for study+delay), study_seq]
        ignore_seq = torch.full((self.seq_len + self.delay,), self.ignore_token)
        y = torch.cat([ignore_seq, study_seq], dim=0)

        return x, y


def get_task_and_dataloader(batch_size=32, seq_len=5, delay=3, vocab_size=10, n_samples=10000):
    task = DelayedCopyTask(vocab_size=vocab_size, seq_len=seq_len, delay=delay, n_samples=n_samples)
    return task, DataLoader(task, batch_size=batch_size, shuffle=True)
