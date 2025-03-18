import preprocess
from preprocess import encode, decode
from preprocess import estimate_loss, estimate_corrector_loss, get_batch, get_corrector_batch
import architecture
import testers

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

# Global hyperparameters
SMALL_ITERS = 1000
LARGE_ITERS = 2000
EVAL_ITERS = 100
CONTEXT_WINDOW_SIZE = 256
PREDICTION_WINDOW_SIZE = 8
NUM_SIMULATIONS = 5

vocab_size = preprocess.vocab_size
device = preprocess.device

torch.manual_seed(305)



# # train predictor 
# predictor = architecture.Predictor(vocab_size, CONTEXT_WINDOW_SIZE).to(device)
# learning_rate = 1e-4
# optimizer = torch.optim.AdamW(predictor.parameters(), lr=learning_rate)

# eval_interval = 200

# loss_list = []

# for it in tqdm(range(LARGE_ITERS)):

#     # every once in a while evaluate the loss on train and val sets
#     if it % eval_interval == 0 or it == LARGE_ITERS - 1:
#         print(f"iteration {it}")
#         losses = estimate_loss(predictor, EVAL_ITERS, CONTEXT_WINDOW_SIZE, device)
#         print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

#     # sample a batch of data
#     xb, yb = get_batch('train', CONTEXT_WINDOW_SIZE, device)

#     # evaluate the loss
#     logits, loss = predictor(xb, yb)
#     loss_list.append(loss.detach().item())
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

# # Check loss convergence
# plt.figure(figsize=(6, 2))
# plt.plot(loss_list)
# plt.xlabel('Iter')
# plt.ylabel('Mean loss')
# plt.savefig('predictor_loss.pdf')


# # train corrector
# corrector = architecture.Corrector(vocab_size, CONTEXT_WINDOW_SIZE, PREDICTION_WINDOW_SIZE).to(device)
# learning_rate = 1e-4
# optimizer = torch.optim.AdamW(corrector.parameters(), lr=learning_rate)

# eval_interval = 200

# loss_list = []

# for it in tqdm(range(LARGE_ITERS)):

#     # every once in a while evaluate the loss on train and val sets
#     if it % eval_interval == 0 or it == LARGE_ITERS - 1:
#         print(f"iteration {it}")
#         losses = estimate_corrector_loss(corrector, EVAL_ITERS, CONTEXT_WINDOW_SIZE, PREDICTION_WINDOW_SIZE, device)
#         print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

#     # sample a batch of data
#     text, index, target = get_corrector_batch('train', CONTEXT_WINDOW_SIZE, PREDICTION_WINDOW_SIZE, device)

#     # evaluate the loss
#     loss = corrector(text, index, target)
#     loss_list.append(loss.detach().item())
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

# # Check loss convergence
# plt.figure(figsize=(6, 2))
# plt.plot(loss_list)
# plt.xlabel('Iter')
# plt.ylabel('Mean loss')
# plt.savefig('corrector_loss.pdf')

# torch.save(predictor.state_dict(), 'predictor.pth')
# torch.save(corrector.state_dict(), 'corrector.pth')


# # predictor = architecture.Predictor(vocab_size, CONTEXT_WINDOW_SIZE).to(device)
# # corrector = architecture.Corrector(vocab_size, CONTEXT_WINDOW_SIZE, PREDICTION_WINDOW_SIZE).to(device)
# # predictor.load_state_dict(torch.load('predictor.pth'))
# # corrector.load_state_dict(torch.load('corrector.pth'))



# # run generator / predictor generate
# contexts = testers.get_tester_batch('test', CONTEXT_WINDOW_SIZE, device, batch_size=100)
# generator = architecture.Generator(vocab_size, predictor, corrector).to(device)
# gen_text = generator(contexts, CONTEXT_WINDOW_SIZE, NUM_SIMULATIONS)
# pred_text = predictor.generate(contexts, CONTEXT_WINDOW_SIZE)

# torch.save(gen_text, 'gen_text.pth')
# torch.save(pred_text, 'pred_text.pth')

# decode(gen_text[0].tolist())
# decode(pred_text[0].tolist())

gen_text = torch.load('gen_text.pth')
pred_text = torch.load('pred_text.pth')


# compare word lengths
gen_length = testers.word_length(gen_text, [int(encode(' ')[0]), int(encode('\n')[0])])
pred_length = testers.word_length(pred_text, [int(encode(' ')[0]), int(encode('\n')[0])])

gen_split = torch.split(torch.sum(gen_length, dim=0)[1:], 5)
pred_split = torch.split(torch.sum(pred_length, dim=0)[1:], 5)


# compare irregular capital letters
capitals = list(range(int(encode('A')[0]), int(encode('Z')[0]+1)))
gen_counts = testers.capital_counts(gen_text, capitals, int(encode('\n')[0]))
pred_counts = testers.capital_counts(pred_text, capitals, int(encode('\n')[0]))

gen_counts = torch.split(torch.bincount(gen_counts, minlength=101)[1:], 5)
pred_counts = torch.split(torch.bincount(pred_counts, minlength=101)[1:], 5)


# loss comparison
gen_x = gen_text[:, -CONTEXT_WINDOW_SIZE:-1]
pred_x = pred_text[:, -CONTEXT_WINDOW_SIZE:-1]
gen_y = gen_text[:, -(CONTEXT_WINDOW_SIZE-1):]
pred_y = pred_text[:, -(CONTEXT_WINDOW_SIZE-1):]

_, gen_loss = predictor(gen_x, gen_y)
_, pred_loss = predictor(pred_x, pred_y)


# results
print("1. Word length dist, 1-150 binned by 5")
print("corrector:", torch.tensor([t.sum() for t in gen_split]))
print("predictor:", torch.tensor([t.sum() for t in pred_split]))
print("/n", "2. Irregualr capital letter counts, 1-100 binned by 5")
print("corrector:", torch.tensor([t.sum() for t in gen_counts]))
print("predictor:", torch.tensor([t.sum() for t in pred_counts]))
print("/n", "3. Mean loss of generated samples")
print("corrector:", gen_loss)
print("predictor:", pred_loss)