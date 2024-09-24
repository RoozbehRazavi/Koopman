import torch
import torch.nn as nn
import unittest
from learn_dynamic import KoopmanMapping

def similarity(z1, z2, temperature = 0.07):
        return torch.exp(torch.dot(z1, z2) / temperature)

class TestLossFunctionEquivalence():
    def __init__(self):
        # Define the dimensions
        self.N = 32  # Batch size
        self.T = 10  # Time steps
        self.d = 128  # State dimension
        self.m = 2  # Action dimension
        self.z_d = 256  # Embedding dimension

        # Mock Koopman operator (using simplified version for testing purposes)
        self.model = KoopmanMapping(obs_dim=self.d, hidden_dim=256, embedding_dim=self.z_d)
        
        # Randomly initialize A and B matrices
        self.A = nn.Parameter(torch.eye(self.z_d))  # Dynamics matrix (d x d)
        self.B = nn.Parameter(torch.randn(self.z_d, self.m))  # Control matrix (d x m)

        # Randomly initialize mock states and actions
        self.states = torch.randn(self.N, self.T, self.d)  # (N x T x d)
        self.actions = torch.randn(self.N, self.T, self.m)  # (N x T x m)

        # Define tolerance for floating-point comparison
        self.tolerance = 1e-6

    def loss_function1(self, A: nn.Parameter, B: nn.Parameter, states, actions):
        """
        Arguments:
            A: Dynamics matrix of size (d x d)
            B: Control matrix of size (d x m)
            states: Batch of initial states of size (N x T x d)
            actions: Actions taken at each time step of size (N x T x m)
            positive_samples: Positive samples (ground truth future states) of size (N x T x d)
            negative_samples: Negative samples (unrelated states) of size (N x T x K x d)
        """
        assert A.requires_grad is False
        assert B.requires_grad is False
        
        N, T, m = actions.shape  # Batch size, time steps, action dimension
        d = states.shape[2]  # State dimension
        
        # Step 1: Compute the anchor (future) state in the embedding space
        z_t = self.model(states)
        
        z_future = torch.zeros((N, T, T, self.z_d))  # Initialize future state embeddings (N x hidden_dim)

        # TODO we should have masking here too!

        for i in range(N):
            for n in range(T):
                for m in range(n+1, T):
                    action = actions[i, n, :].unsqueeze(0)
                    if m == n+1:
                        z_t_ = z_t[i, n, :].unsqueeze(-1)
                        temp = A @ z_t_ + B @ action.T
                        z_future[i, n, m] = temp.squeeze(-1)
                    else:
                        temp = A @ z_future[i, n, m-1, :].unsqueeze(-1) + B @ action.T
                        z_future[i, n, m] = temp.squeeze(-1)

        # Pass positive samples through the RNN and get embeddings
        hidden_pos,_  = self.model.rnn(states)  # (1 x N x hidden_dim)
        z_positive = self.model.pos_neg_embedding_head(hidden_pos)  # (N x embedding_dim)
        loss = 0
        for i in range(N):
            for n in range(T):
                for m in range(n, T):
                    if m > n:
                        positive_similarity = similarity(z_future[i, n, m], z_positive[i, m])
                        negative_similarity = 0
                        for k in range(T):
                            if k != m:
                                negative_similarity += similarity(z_future[i, n, m], z_positive[i, k])
                        loss += positive_similarity / (negative_similarity + positive_similarity)
        
        # TODO what is the purpose of this one? Log-softmax over the positive and negative similarities
        # logits = torch.cat([positive_similarity.unsqueeze(1), negative_similarity], dim=1)  # (N x (1 + K))
        # labels = torch.zeros(N, dtype=torch.long).to(logits.device)  # Positive class index is 0
        # loss = nn.CrossEntropyLoss()(logits, labels)
        
        return loss

    def vectorized_loss_function(self, A, B, states, actions):
        """Vectorized version of the loss function."""
        N, T, d = states.shape  # Batch size, time steps, state dimension
        m = actions.shape[-1]  # Action dimension

        z_t = self.model(states)  # (N x T x d) - Embedding of initial states

        A_expanded = A.unsqueeze(0).expand(N, -1, -1)  # (N x d x d) -> to apply to all samples in batch
        B_expanded = B.unsqueeze(0).expand(N, -1, -1)  # (N x d x m) -> to apply to all samples in batch

        z_future = torch.zeros_like(z_t)  # (N x T x d)

        z_future[:, 0, :] = A_expanded @ z_t[:, 0, :].unsqueeze(-1) + (B_expanded @ actions[:, 0, :].unsqueeze(-1)).squeeze(-1)

        for t in range(1, T):
            z_future[:, t, :] = A_expanded @ z_future[:, t-1, :].unsqueeze(-1) + (B_expanded @ actions[:, t, :].unsqueeze(-1)).squeeze(-1)

        _, hidden_pos = self.model.rnn(states)  # (1 x N x hidden_dim)
        z_positive = self.model.pos_neg_embedding_head(hidden_pos.squeeze(0))  # (N x embedding_dim)

        positive_similarity = torch.einsum('ntd,nd->nt', z_future, z_positive) / (
                torch.norm(z_future, dim=-1) * torch.norm(z_positive, dim=-1, keepdim=True)
        )

        z_positive_expanded = z_positive.unsqueeze(1).expand(N, T, d)  # (N x T x d)
        z_future_expanded = z_future.unsqueeze(2).expand(N, T, T, d)  # (N x T x T x d)

        negative_similarity = torch.einsum('ntd,ntd->nt', z_future_expanded, z_positive_expanded) / (
            torch.norm(z_future, dim=-1) * torch.norm(z_positive, dim=-1, keepdim=True)
        )

        temperature = 0.07
        positive_similarity = positive_similarity / temperature
        negative_similarity = negative_similarity / temperature

        loss = (positive_similarity / negative_similarity).mean()

        return loss

    def test_loss_function_equivalence(self):
        """Test that the original and vectorized loss functions produce the same result."""
        self.A.requires_grad = False
        self.B.requires_grad = False
        original_loss = self.loss_function1(self.A, self.B, self.states, self.actions)
        vectorized_loss = self.vectorized_loss_function(self.A, self.B, self.states, self.actions)

        self.assertAlmostEqual(original_loss.item(), vectorized_loss.item(), delta=self.tolerance, 
                               msg="Loss function outputs are not equal within tolerance")


if __name__ == '__main__':
    test = TestLossFunctionEquivalence()
    test.test_loss_function_equivalence()
