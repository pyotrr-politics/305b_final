import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from tqdm import tqdm
from preprocess import device

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, context_window_size, num_heads, head_size, embed_size=384):
        """
        Args:
            context_window_size: int, number of tokens considered in the past for attention (T)
            num_heads: int, number of heads (H)
            head_size: int, size of the head embedding dimension
            embed_size: int, size of the token embedding dimension
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.context_window_size = context_window_size
        self.key = nn.ModuleList([nn.Linear(embed_size, head_size, bias=False) \
                            for _ in range(num_heads)])
        self.query = nn.ModuleList([nn.Linear(embed_size, head_size, bias=False) \
                            for _ in range(num_heads)])
        self.value = nn.ModuleList([nn.Linear(embed_size, head_size, bias=False) \
                            for _ in range(num_heads)])
        self.output = nn.ModuleList([nn.Linear(head_size, embed_size, bias=False) \
                            for _ in range(num_heads)])
    
        pass

    def forward(self, x):
        head_outputs = torch.zeros_like(x)
        for i in range(self.num_heads):            
            attn_weights = torch.einsum('bij, bkj -> bik', self.query[i](x), self.key[i](x))
            tril = torch.tril(torch.ones(attn_weights.shape)).to(device)
            attn_weights = (attn_weights * tril)/math.sqrt(self.head_size)
            attn_weights = F.softmax(attn_weights, dim=-1)
            head_outputs = head_outputs + self.output[i](attn_weights @ self.value[i](x))

        return head_outputs
    



class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity
        Given to you, you don't need to write any code here!
    """

    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )

    def forward(self, x):
        return self.net(x)
    

class DeepFeedForward(nn.Module):
    """ two non-linear layers
    """

    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )

    def forward(self, x):
        return self.net(x)



class TransformerBlock(nn.Module):
    """ Transformer block: communication across sequence length, followed by communication across embedding space
        Uses multi-headed attention
    """

    def __init__(self, vocab_size, context_window_size, embed_size=384, num_heads=6):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

        self.feed_forward = FeedForward(embed_size)
        self.head_size = embed_size // num_heads
        self.atten_heads = MultiHeadAttention(context_window_size, num_heads, self.head_size, embed_size=384)
        
    def forward(self, x):
        x = x + self.atten_heads(self.ln1(x)) # communication over sequence length
        x = x + self.feed_forward(self.ln2(x)) # communication across embedding space
        return x



class Predictor(nn.Module):
    def __init__(self, vocab_size, context_window_size, embed_size=384, num_heads=6, n_layers=6):
        """
          Args:
              vocab_size: int, number of tokens in the vocabulary (V)
              context_window_size: int, size of the context window (T)
              embed_size: int, embedding size (D)
              num_heads: int, number of heads (H)
              n_layers: int, number of layers (M)
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(context_window_size, embed_size)
        self.blocks = nn.Sequential(*[
            TransformerBlock(vocab_size,
                             context_window_size,
                             embed_size=embed_size,
                             num_heads=num_heads)
            for _ in range(n_layers)])

        # final layer norm
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)

        # good initialization
        self.apply(self._init_weights)
        self.context_window_size = context_window_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, targets=None):
        """
        Agrgs:
            token_ids: tensor of integers, provides the contet, shape (B, T)
            targets: tensor of integers, provides the tokens we are preidcitng, shape (B, T)
        """
        B, T = token_ids.shape

        # token_ids and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(token_ids) # (B, T, D)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, D)
        x = tok_emb + pos_emb # (B, T, D)

        logits = self.lm_head(self.ln_f(self.blocks(x)))
        if targets is None:
            loss = None
        else:
            loss = torch.logsumexp(logits, dim=2) - \
                   logits[torch.arange(logits.shape[0]).unsqueeze(1),
                          torch.arange(logits.shape[1]),
                          targets]
            loss = torch.mean(loss)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens):
        """
        Args:
            token_ids: tensor of integers forming the context, shape (B, T)
            max_new_tokens: int, max number of tokens to generate
        """
        B, T = token_ids.shape
        for j in range(max_new_tokens):
            if self.context_window_size >= T+j:
                ref = 0
            else:
                ref = T+j-self.context_window_size
            logits, _ = self.forward(token_ids[:, ref:])
            pred = torch.distributions.Categorical(F.softmax(logits[:, -1], dim=-1)).sample().unsqueeze(1)
            token_ids = torch.hstack([token_ids, pred.to(device)])

        return token_ids[:, -max_new_tokens:]

    




class Corrector(nn.Module):

    def __init__(self, vocab_size, context_window_size, prediction_size, 
                 embed_size=384, num_heads=6, n_layers=6):
        """
          Args:
              vocab_size: int, number of tokens in the vocabulary (V)
              context_window_size: int, size of the context window (T)
              prediction_size: int, size of the prediction window (k)
              embed_size: int, embedding size (D)
              num_heads: int, number of heads (H)
              n_layers: int, number of layers (M)
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(context_window_size, embed_size)
        self.blocks = nn.Sequential(*[
            TransformerBlock(vocab_size,
                             context_window_size,
                             embed_size=embed_size,
                             num_heads=num_heads)
            for _ in range(n_layers)])
        self.feed_forward = DeepFeedForward(embed_size)

        # final layer norm
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)

        # good initialization
        self.apply(self._init_weights)

        self.context_window_size = context_window_size
        self.prediction_size = prediction_size
        self.m = nn.Softplus()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, index, targets=None):
        """
        Agrgs:
            token_ids: tensor of integers, provides the contet, shape (B, T)
            index: tensor of integers, shape (I, )
            targets: tensor of integers, shape (B, I, V)
        """
        B, T = token_ids.shape
        # token_ids and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(token_ids) # (B, T, D)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, D)
        x = tok_emb + pos_emb # (B, T, D)

        if isinstance(index, int):
            x = self.blocks(x)[:, index, :].unsqueeze(1)
        else:
            x = self.blocks(x)[:, index, :]
        lambdas = self.lm_head(self.ln_f(self.feed_forward(x)))
        lambdas = self.m(lambdas)
        if targets is None:
            loss = None
        else:
            # write Poisson loss function
            poisson_loss = nn.PoissonNLLLoss(log_input=False, reduction='none')
            loss =  torch.sum(poisson_loss(lambdas, targets), dim=2) - \
                    poisson_loss(torch.sum(lambdas, dim=2), torch.tensor(self.prediction_size))
        return torch.mean(loss)




class Generator(nn.Module):
    def __init__(self, vocab_size, predictor: Predictor, corrector: Corrector):
        super().__init__()
        self.vocab_size = vocab_size
        self.predictor = predictor
        self.corrector = corrector
        self.context_window_size = corrector.context_window_size
        self.prediction_size = corrector.prediction_size

    @torch.no_grad()
    def forward(self, token_ids, max_new_tokens, num_simulation):
        """
        Args:
            token_ids: tensor of integers forming the context, shape (B, T)
            max_new_tokens: int, max number of tokens to generate
            num_simulation: int, number of cadnidate predictions to simulate (M)
        """
        B, T = token_ids.shape
        num_blocks = max_new_tokens // self.prediction_size
        
        for _ in tqdm(range(num_blocks)):
            ref = token_ids
            if self.context_window_size < ref.shape[1]:
                ref = ref[:, -self.context_window_size:]
            
            new_pred = torch.ones(1, device=device)
            target = torch.ones((B, 1, self.vocab_size)).to(device)
            loss = self.corrector(ref, ref.shape[1]-1, target)
            
            for _ in range(num_simulation):
                ref = token_ids
                for _ in range(self.prediction_size):
                    if self.context_window_size < ref.shape[1]:
                        ref = ref[:, -self.context_window_size:]
                    logits, _ =  self.predictor(ref)
                    pred = torch.distributions.Categorical(F.softmax(logits[:, -1], dim=-1)).sample().unsqueeze(1)
                    ref = torch.hstack([ref, pred.to(device)])

                pred = ref[:, -self.prediction_size:]
                target = []
                for k in range(B):
                    target.append(torch.bincount(pred[k], minlength=self.vocab_size).unsqueeze(0))
                
                ref = ref[:, :-self.prediction_size]
                loss_temp = self.corrector(ref, ref.shape[1]-1, torch.stack(target))
                if loss > loss_temp:
                    loss = loss_temp
                    new_pred = pred
            
            token_ids = torch.cat([token_ids, new_pred], dim=1)

        for j in range(max_new_tokens % self.prediction_size):
            if self.context_window_size >= T+j:
                ref = 0
            else:
                ref = T+j-self.context_window_size
            logits, _ = self.predictor(token_ids[:, ref:])
            pred = torch.distributions.Categorical(F.softmax(logits[:, -1], dim=-1)).sample().unsqueeze(1)
            token_ids = torch.hstack([token_ids, pred.to(device)])

        return token_ids[:, -max_new_tokens:]
    
