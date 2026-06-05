"""
Music Transformer: A decoder-only Transformer model for music generation.

Uses relative positional encoding (Music Transformer style) to capture
musical structure and long-range dependencies.

Based on:
- "Music Transformer" (Huang et al., 2018)
- "Attention Is All You Need" (Vaswani et al., 2017)
"""
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for music transformer.
    Computes position bias scores for each attention head.
    """
    
    def __init__(self, d_model: int, max_relative_position: int, n_heads: int):
        super().__init__()
        self.max_relative_position = max_relative_position
        self.n_heads = n_heads
        
        # Embedding for relative positions
        self.embedding = nn.Embedding(2 * max_relative_position + 1, n_heads)
        
    def forward(self, length: int, device: torch.device):
        """
        Compute relative position bias.
        
        Args:
            length: Sequence length
            device: Target device
        
        Returns:
            bias: (1, n_heads, length, length) position bias
        """
        # Create relative position matrix
        range_vec = torch.arange(length, device=device)
        relative_positions = range_vec[None, :] - range_vec[:, None]  # (L, L)
        
        # Clamp to valid range
        relative_positions = torch.clamp(
            relative_positions, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # Shift to [0, 2*max_relative_position]
        relative_positions += self.max_relative_position
        
        # Get bias (L, L, n_heads) then permute to (1, n_heads, L, L)
        bias = self.embedding(relative_positions)  # (L, L, n_heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, n_heads, L, L)
        
        return bias


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE) - a modern alternative to relative positions.
    Encodes position by rotating query and key vectors.
    """
    
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        
        # Precompute the rotation frequencies
        theta = 10000.0 ** (-torch.arange(0, d_model, 2).float() / d_model)
        position = torch.arange(max_len).float().unsqueeze(1)
        freqs = position * theta.unsqueeze(0)  # (max_len, d_model/2)
        self.register_buffer("freqs", freqs, persistent=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary encoding to queries or keys.
        
        Args:
            x: (batch_size, n_heads, seq_len, d_head)
        
        Returns:
            Rotated tensor of same shape
        """
        seq_len = x.size(2)
        d_head = x.size(3)
        d_half = d_head // 2
        
        freqs = self.freqs[:seq_len, :d_half]  # (seq_len, d_half)
        
        # Reshape x into pairs
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], d_half, 2))
        freqs_complex = torch.view_as_complex(freqs.unsqueeze(0).unsqueeze(0).expand(
            x.size(0), x.size(1), -1, -1
        ))
        
        # Rotate
        rotated = torch.view_as_real(x_complex * freqs_complex).reshape(*x.shape)
        return rotated.type_as(x)


class MultiHeadRelativeAttention(nn.Module):
    """
    Multi-head self-attention with relative positional bias.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 max_relative_position: int = 128, use_rope: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_rope = use_rope
        self.scale = math.sqrt(self.d_head)
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        if use_rope:
            self.rope = RotaryPositionalEncoding(d_model, max_relative_position * 2)
            self.pos_bias = None
        else:
            self.pos_bias = RelativePositionalEncoding(d_model, max_relative_position, n_heads)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len) or (seq_len, seq_len) - optional attention mask
        
        Returns:
            (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape for multi-head
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Apply rotary position encoding if using RoPE
        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, L, L)
        
        # Add relative positional bias (if not using RoPE)
        if not self.use_rope and self.pos_bias is not None:
            pos_bias = self.pos_bias(seq_len, x.device)  # (1, H, L, L)
            attn_scores = attn_scores + pos_bias
        
        # Apply mask (causal masking for decoder)
        if mask is not None:
            if mask.dim() == 2:
                # (B, L) -> (B, 1, 1, L)
                mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Causal mask (prevent attending to future positions)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax and apply dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum
        context = torch.matmul(attn_weights, v)  # (B, H, L, d_head)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(context)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """A single transformer decoder block."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 max_relative_position: int = 128, use_rope: bool = False):
        super().__init__()
        self.attention = MultiHeadRelativeAttention(
            d_model, n_heads, dropout, max_relative_position, use_rope
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attention(x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x


class MusicTransformer(nn.Module):
    """
    Decoder-only Music Transformer model.
    
    Generates music token sequences autoregressively.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_layers: int = 6,
                 n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1,
                 max_seq_len: int = 2048, max_relative_position: int = 128,
                 use_rope: bool = False):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Absolute positional embedding (used if not using RoPE)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_relative_position, use_rope)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        # Share weights between input and output embeddings
        self.token_embedding.weight = self.output_proj.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier uniform."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len) token IDs
            mask: (batch_size, seq_len) padding mask
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)  # (B, L, d_model)
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm and output projection
        x = self.ln_final(x)
        logits = self.output_proj(x)
        
        return logits
    
    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 512,
                 temperature: float = 1.0, top_k: int = 40, top_p: float = 0.9,
                 eos_token_id: int = 2, pad_token_id: int = 0) -> torch.Tensor:
        """
        Generate a sequence autoregressively.
        
        Args:
            prompt: (1, prompt_len) initial token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering
            top_p: Nucleus (top-p) filtering
            eos_token_id: End-of-sequence token ID
            pad_token_id: Padding token ID
        
        Returns:
            (1, total_len) generated token IDs including prompt
        """
        self.eval()
        device = prompt.device
        generated = prompt.clone()
        
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            if generated.size(1) > self.max_seq_len:
                generated = generated[:, -self.max_seq_len:]
            
            # Forward pass
            logits = self.forward(generated)
            next_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(next_logits, top_k, dim=-1)
                min_top_k = top_k_values[:, -1].unsqueeze(-1)
                next_logits[next_logits < min_top_k] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS
            if next_token.item() == eos_token_id:
                break
        
        return generated
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
