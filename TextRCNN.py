import torch.nn as nn
import torch
import torch.nn.functional as F


class TextRCNN(nn.Module):
    def __init__(self):
        super(TextRCNN, self).__init__()
        self.lstm = nn.LSTM(100, 128, 2, bidirectional=True, batch_first=True, dropout=0.05)
        self.maxpool = nn.MaxPool1d(39)
        self.fc = nn.Linear(128 * 2 + 100, 6)

    def forward(self, x):
        # [batch_size, seq_len, embeding]=[16, 27, 100]
        out, _ = self.lstm(x)
        out = torch.cat((x, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out
