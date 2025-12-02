# models/ai_audio_eq.py
import torch
import torch.nn as nn
class AudioEQModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20,64),
            nn.ReLU(),
            nn.Linear(64,31)
        )
    def forward(self,x):
        return self.net(x)
if __name__ == "__main__":
    m = AudioEQModel()
    torch.save(m.state_dict(), "ai_audio_eq.pt")