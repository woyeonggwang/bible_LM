import torch
from model import LanguageModel, encode, decode, device, ckpt_path
model = LanguageModel().to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))

model.eval()
prompt = "In the begining"                                   
idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

with torch.no_grad():                                     
    out = model.generate(idx, max_new_tokens=500)         

print(decode(out[0].tolist()))  