import string
import random
import numpy as np
import torch


def precond_update_prob_schedule(
    max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500
):
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at 1.0 for 500 steps then exponentially anneal down to
    `min_prob` by 4000 steps. Default settings work very well for most models and
    training regimes.
    """
    max_prob_ = torch.tensor(max_prob, dtype=torch.float32)
    min_prob_ = torch.tensor(min_prob, dtype=torch.float32)
    decay_ = torch.tensor(decay, dtype=torch.float32)
    flat_start_ = torch.tensor(flat_start, dtype=torch.float32)

    @torch.compile
    def _schedule(n):
        """Exponential anneal with flat start."""
        prob = max_prob_ * torch.exp(-decay_ * (n - flat_start_))
        prob.clamp_(min=min_prob_, max=max_prob_)
        return prob

    return _schedule


def compute_udelle_update(Q, p, pre_grad, debiased_momentum, loss_fn, precond_dtype, device):
    """
    Abstracts Udell's hypergradient surrogate computation.
    
    This function computes the hypergradient of:
      h(P) = [f(x - P∇f(x)) - f(x)] / (∥∇f(x)∥² + eps)
    with respect to each factor Q (where P = QᵀQ).
    
    Args:
      Q (list of torch.Tensor): List of preconditioner factors.
      p (torch.Tensor): The current parameter tensor.
      pre_grad (torch.Tensor): Preconditioned gradient (i.e. P∇f(x)).
      debiased_momentum (torch.Tensor): The momentum (∇f(x)) used for normalization.
      loss_fn (callable): A function that takes a parameter tensor and returns a scalar loss.
      precond_dtype (torch.dtype): Dtype for preconditioner.
      device (torch.device): Device on which computations occur.
      
    Returns:
      List of hypergradient tensors, one for each Q.
    """
    # Enable gradient tracking on Q
    for q in Q:
        q.requires_grad_(True)
    p_candidate = p - pre_grad
    loss_current = loss_fn(p)
    loss_candidate = loss_fn(p_candidate)
    eps = torch.tensor(1e-8, dtype=precond_dtype, device=device)
    h_val = (loss_candidate - loss_current) / (debiased_momentum.norm()**2 + eps)
    hyper_updates = []
    for q in Q:
        grad_q = torch.autograd.grad(h_val, q, retain_graph=True)[0]
        hyper_updates.append(grad_q)
    for q in Q:
        q.requires_grad_(False)
    return hyper_updates


class Kron(torch.optim.Optimizer):
    """Implements PSGD Kron from https://github.com/lixilinx/psgd_torch,
    now augmented to allow a composite update:
    
      Composite update = (Criterion-3 update) + (udelle_weight * Udell hypergradient update)
    
    Criterion-3 is based on minimizing
        c₃(P) = E[δgᵀ P δg + δθᵀ P⁻¹ δθ],
    and Udell's hypergradient surrogate is based on
        h(P) = [f(x - P∇f(x)) - f(x)] / ∥∇f(x)∥².
    
    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float): Learning rate.
        b1 (float): Momentum parameter.
        weight_decay (float): Weight decay.
        preconditioner_update_probability (callable or float, optional): Probability of
            updating the preconditioner. If None, defaults to a schedule that anneals
            from 1.0 to 0.03 by 4000 steps.
        max_size_triangular (int): Max size for dim's preconditioner to be triangular.
        min_ndim_triangular (int): Minimum number of dimensions a layer needs
            to have triangular preconditioners.
        memory_save_mode (str, optional): One of [None, 'smart_one_diag', 'one_diag', 'all_diag'].
        momentum_into_precond_update (bool): Whether to send momentum into preconditioner
            update instead of raw gradients.
        precond_lr (float): Learning rate for preconditioner.
        precond_init_scale (float): Initial scale for preconditioners.
        mu_dtype (torch.dtype, optional): Dtype of the momentum accumulator.
        precond_dtype (torch.dtype, optional): Dtype of the preconditioner.
        loss_fn (callable, optional): A function that computes the loss f(.) given a parameter tensor.
            This is used to compute Udell's hypergradient surrogate.
        udelle_weight (float): Lambda weight for Udell's hypergradient surrogate update.
            Final update will be: criterion3_update + udelle_weight * udelle_update.
    """

    def __init__(
        self,
        params,
        lr=0.0003,
        b1=0.9,
        weight_decay=0.0,
        preconditioner_update_probability=None,
        max_size_triangular=8192,
        min_ndim_triangular=2,
        memory_save_mode=None,
        momentum_into_precond_update=True,
        precond_lr=0.1,
        precond_init_scale=1.0,
        mu_dtype=None,
        precond_dtype=None,
        loss_fn=None,         # For computing f(.) used in Udell's hypergradient surrogate.
        udelle_weight=0.0     # Weight (lambda) for Udell's surrogate update.
    ):
        if preconditioner_update_probability is None:
            preconditioner_update_probability = precond_update_prob_schedule()

        defaults = dict(
            lr=lr,
            b1=b1,
            weight_decay=weight_decay,
            preconditioner_update_probability=preconditioner_update_probability,
            max_size_triangular=max_size_triangular,
            min_ndim_triangular=min_ndim_triangular,
            memory_save_mode=memory_save_mode,
            momentum_into_precond_update=momentum_into_precond_update,
            precond_lr=precond_lr,
            precond_init_scale=precond_init_scale,
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
        )
        super(Kron, self).__init__(params, defaults)

        self._prob_step = torch.tensor(0, dtype=torch.int32)
        self._update_counter = torch.tensor(0, dtype=torch.int32)
        self.rng = random.Random(42)
        self.loss_fn = loss_fn  # Loss function for Udell's surrogate (if provided).
        self.udelle_weight = udelle_weight  # Weight (lambda) for Udell's hypergradient update.

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        total_momentum_size = 0
        total_momentum_mb = 0
        total_precond_size = 0
        total_precond_mb = 0

        # Update preconditioners deterministically.
        update_prob = self.param_groups[0]["preconditioner_update_probability"]
        if callable(update_prob):
            update_prob = update_prob(self._prob_step.to(dtype=torch.float32))
        self._update_counter += 1
        do_update = self._update_counter >= 1 / update_prob
        if do_update:
            self._update_counter = torch.tensor(0, dtype=torch.int32)
        self._prob_step += 1

        # Balance preconditioners roughly every 100 updates.
        balance = self.rng.random() < 0.01 and do_update

        for group in self.param_groups:
            mu_dtype = group.get("mu_dtype")
            precond_dtype = group.get("precond_dtype", torch.float32)
            if precond_dtype is None:
                precond_dtype = torch.float32
            momentum_into_precond_update = group.get("momentum_into_precond_update", True)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=mu_dtype or p.dtype)
                    state["Q"], state["exprs"] = _init_Q_exprs(
                        p,
                        group["precond_init_scale"],
                        group["max_size_triangular"],
                        group["min_ndim_triangular"],
                        group["memory_save_mode"],
                        dtype=precond_dtype,
                    )

                    momentum_size = state["momentum_buffer"].numel()
                    momentum_mb = momentum_size * state["momentum_buffer"].element_size() / (2**20)
                    total_momentum_size += momentum_size
                    total_momentum_mb += momentum_mb

                    precond_size = sum(q.numel() for q in state["Q"])
                    precond_mb = sum(q.numel() * q.element_size() for q in state["Q"]) / (2**20)
                    total_precond_size += precond_size
                    total_precond_mb += precond_mb

                state["step"] += 1

                beta = group["b1"]
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(beta).add_(grad, alpha=1 - beta)
                # restore momentum dtype
                if mu_dtype is not None:
                    momentum_buffer.copy_(momentum_buffer.to(dtype=mu_dtype))
                debiased_momentum = momentum_buffer / (1 - beta ** state["step"])
                debiased_momentum = debiased_momentum.to(dtype=precond_dtype)

                # Balance preconditioners about every 100 updates.
                if grad.dim() > 1 and balance:
                    _balance_Q(state["Q"])

                extra_update = None
                # --- Udell's hypergradient surrogate computation ---
                # If a loss function is provided and udelle_weight > 0, compute the extra update.
                if self.loss_fn is not None and self.udelle_weight != 0.0:
                    # Compute the preconditioned gradient and candidate update.
                    pre_grad = _precond_grad(state["Q"], state["exprs"], debiased_momentum)
                    p_candidate = p - pre_grad
                    # Compute loss on current parameters and candidate update.
                    loss_current = self.loss_fn(p)
                    loss_candidate = self.loss_fn(p_candidate)
                    eps = torch.tensor(1e-8, dtype=precond_dtype, device=grad.device)
                    # h(P) = [f(x - P∇f(x)) - f(x)] / ∥∇f(x)∥²  -> Udell's criterion.
                    h_val = (loss_candidate - loss_current) / (debiased_momentum.norm()**2 + eps)
                    # Abstract Udell's update into a separate function.
                    hyper_updates = compute_udelle_update(
                        state["Q"], p, pre_grad, debiased_momentum, self.loss_fn, precond_dtype, grad.device
                    )
                    # Scale the hypergradient by udelle_weight.
                    extra_update = [group["precond_lr"] * self.udelle_weight * gu for gu in hyper_updates]
                # --- End Udell's hypergradient surrogate computation ---

                # --- Criterion-3 update ---
                # The following _update_precond call computes the gradient of
                # c₃(P) = E[δgᵀ P δg + δθᵀ P⁻¹ δθ] (this is our original criterion-3).
                # If extra_update is provided (from Udell's surrogate), it is added.
                _update_precond(
                    state["Q"],
                    state["exprs"],
                    debiased_momentum if momentum_into_precond_update else grad.to(dtype=precond_dtype),
                    torch.tensor(group["precond_lr"], dtype=precond_dtype, device=grad.device),
                    torch.tensor(torch.finfo(precond_dtype).tiny, dtype=precond_dtype, device=grad.device),
                    extra_update  # Composite update: criterion-3 + udelle hypergradient weighted by udelle_weight.
                )
                # --- End Criterion-3 update ---

                # Precondition gradients.
                pre_grad = _precond_grad(state["Q"], state["exprs"], debiased_momentum)

                # Clip update RMS.
                _clip_update_rms(pre_grad)

                # Apply weight decay and update parameters.
                if group["weight_decay"] != 0 and p.dim() >= 2:
                    pre_grad.add_(p, alpha=group["weight_decay"])
                p.add_(pre_grad.to(dtype=p.dtype), alpha=-group["lr"])

        if total_momentum_size > 0:
            print(
                f"PSGD Momentum buffer size: {total_momentum_size} "
                f"elements, {total_momentum_mb:.2f} MB"
            )
            print(
                f"PSGD Preconditioners size: {total_precond_size} "
                f"elements, {total_precond_mb:.2f} MB"
            )

        return loss


def _init_Q_exprs(t, scale, max_size, min_ndim_triangular, memory_save_mode, dtype=None):
    """For a scalar or tensor t, we initialize its preconditioner Q and
    reusable einsum expressions for updating Q and preconditioning gradient.
    """
    letters = string.ascii_lowercase + string.ascii_uppercase

    dtype = dtype if dtype is not None else t.dtype
    shape = t.shape
    if len(shape) == 0:  # scalar
        Q = [scale * torch.ones_like(t, dtype=dtype)]
        exprA = ",->"
        exprGs = [",->"]
        exprP = ",,->"
    else:  # tensor
        if len(shape) > 13:
            raise ValueError(
                f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!"
            )

        scale = scale ** (1 / len(shape))

        if memory_save_mode is None:
            dim_diag = [False for _ in shape]
        elif memory_save_mode == "smart_one_diag":
            rev_sorted_dims = np.argsort(shape)[::-1]
            dim_diag = [False for _ in shape]
            sorted_shape = sorted(shape)
            if len(shape) > 1 and sorted_shape[-1] > sorted_shape[-2]:
                dim_diag[rev_sorted_dims[0]] = True
        elif memory_save_mode == "one_diag":
            rev_sorted_dims = np.argsort(shape)[::-1]
            dim_diag = [False for _ in shape]
            dim_diag[rev_sorted_dims[0]] = True
        elif memory_save_mode == "all_diag":
            dim_diag = [True for _ in shape]
        else:
            raise ValueError(
                f"Invalid memory_save_mode: {memory_save_mode}, must be one of "
                "[None, 'smart_one_diag', 'one_diag', 'all_diag']"
            )

        Q = []
        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
        for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
            if (
                size == 1
                or size > max_size
                or len(shape) < min_ndim_triangular
                or dim_d
            ):
                # use diagonal matrix as preconditioner for this dim
                Q.append(scale * torch.ones(size, dtype=dtype, device=t.device))

                piece1A.append(letters[i])
                piece2A = piece2A + letters[i]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [
                        (letters[i + 13] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                subscripts = piece1 + "," + piece1 + "->" + letters[i + 13]
                exprGs.append(subscripts)

                piece1P.append(letters[i + 13])
                piece2P.append(letters[i + 13])
                piece3P = piece3P + letters[i + 13]
                piece4P = piece4P + letters[i + 13]
            else:
                # use triangular matrix as preconditioner for this dim
                Q.append(scale * torch.eye(size, dtype=dtype, device=t.device))

                piece1A.append(letters[i] + letters[i + 13])
                piece2A = piece2A + letters[i + 13]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [
                        (letters[i + 13] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                piece2 = "".join(
                    [
                        (letters[i + 26] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                subscripts = (
                    piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26]
                )
                exprGs.append(subscripts)

                a, b, c = (letters[i], letters[i + 13], letters[i + 26])
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

        exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprP = (
            ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
        )

    exprGs = tuple(exprGs)
    return [Q, (exprA, exprGs, exprP)]


@torch.compile
def _balance_Q(Q_in):
    norms = torch.stack([q.norm(float("inf")) for q in Q_in])
    geometric_mean = norms.log().mean().exp()
    norms = geometric_mean / norms
    torch._foreach_mul_(Q_in, list(norms))


def _lb(A: torch.Tensor, max_abs: torch.Tensor):
    """Cheap lower bound for the spectral norm of A."""
    A /= max_abs
    a0 = torch.einsum("ij,ij->j", A, A)
    i = torch.argmax(a0)
    x = torch.index_select(A, 1, i).flatten().contiguous()
    x = torch.einsum("i,ij->j", x, A)
    x /= x.norm()
    x = torch.einsum("j,kj->k", x, A)
    x = x.norm()
    x *= max_abs
    return x


def _solve_triangular_right(X: torch.Tensor, A: torch.Tensor):
    """X @ inv(A)"""
    orig_dtype = A.dtype
    return (
        torch.linalg.solve_triangular(
            A.float(),
            X.reshape(-1, X.size(-1)).float(),
            upper=True,
            left=False,
            unitriangular=False,
        )
        .to(dtype=orig_dtype)
        .reshape_as(X)
    )


def _calc_A_and_conjB(exprA, G, Q):
    """Calculate A and conjB."""
    order = G.dim()
    V = torch.randn_like(G)
    eps = torch.tensor(torch.finfo(torch.float32).eps, dtype=G.dtype, device=G.device)
    G += eps.sqrt() * G.abs().mean() * V
    conjB = V.permute(*range(1, order), 0)
    for i, q in enumerate(Q):
        conjB = conjB / q if q.dim() < 2 else _solve_triangular_right(conjB, q)
        if i < order - 1:
            conjB = torch.transpose(conjB, i, order - 1)
    A = torch.einsum(exprA, *Q, G)
    return A, conjB


@torch.compile
def _update_precond(Q, exprs, G, step, tiny, extra_update=None):
    """
    Update the preconditioner on the Lie group.

    Here, Q is the Cholesky factor such that P = QᵀQ, and Q belongs
    to the Lie group of upper triangular matrices with positive diagonal elements.
    
    The standard update (criterion-3 update) is computed via:
       c₃(P) = E[δgᵀ P δg + δθᵀ P⁻¹ δθ]
    whose gradient (relative gradient on Q) is computed using torch.einsum.
    
    If extra_update is provided (computed from Udell's hypergradient surrogate),
    it is added to the relative gradient update.
    
    Finally, Q is updated by subtracting the composite update:
       Q ← Q - step * [grad_criterion3 + extra_update]
    which performs a relative update in the Lie algebra of Q.
    """
    exprA, exprGs, _ = exprs
    A, conjB = _calc_A_and_conjB(exprA, G, Q)
    for q, exprG in zip(Q, exprGs):
        # Compute the relative gradient update for criterion-3.
        term1 = torch.einsum(exprG, A, A)
        term2 = torch.einsum(exprG, conjB, conjB)
        term1, term2 = term1 - term2, term1 + term2  # This is the criterion-3 gradient.
        term1 *= step
        norm = term2.norm(float("inf"))
        if q.dim() < 2:
            term1 *= q / norm.clamp_(min=tiny)
        else:
            torch.triu(term1, out=term1)
            term1 /= torch.where(norm > 0, _lb(term2, norm), norm).clamp_(min=tiny)
            term1 = torch.mm(term1, q)
        # Add the extra update from Udell's hypergradient surrogate if provided.
        if extra_update is not None:
            term1 = term1 + extra_update.pop(0)
        # Update Q on the Lie group.
        q.sub_(term1)

@torch.compile
def _precond_grad(Q, exprs, G):
    """Precondition gradient G with preconditioner Q."""
    return torch.einsum(exprs[-1], *Q, *Q, G)


@torch.compile
def _clip_update_rms(g):
    g.mul_(
        torch.minimum(
            torch.tensor(1.0, dtype=g.dtype, device=g.device),
            1.1 / g.square().mean().sqrt().add(1e-12),
        )
    )
