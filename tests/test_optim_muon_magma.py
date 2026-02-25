import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.optim import MuonAdamW


def test_muon_magma_updates_momentum_on_masked_steps():
    p = torch.nn.Parameter(torch.tensor([[1.0, -2.0, 3.0], [0.5, -0.25, 1.5]], dtype=torch.float32))
    opt = MuonAdamW([
        {
            "kind": "muon",
            "params": [p],
            "lr": 1e-2,
            "momentum": 0.95,
            "ns_steps": 2,
            "beta2": 0.95,
            "weight_decay": 0.0,
            "magma": True,
            "survival_prob": 0.5,
            "temperature": 2.0,
            "ema_decay": 0.9,
        }
    ])

    torch.manual_seed(1234)
    found_masked_step = False
    for _ in range(64):
        p.grad = torch.tensor([[0.4, -0.2, 0.8], [0.1, -0.3, 0.5]], dtype=torch.float32)
        p_before = p.detach().clone()
        state = opt.state[p]
        mom_before = state.get("momentum_buffer", torch.zeros(1, *p.shape, dtype=p.dtype, device=p.device)).detach().clone()
        opt.step()
        mom_after = opt.state[p]["momentum_buffer"]

        if torch.equal(p, p_before):
            found_masked_step = True
            assert not torch.allclose(mom_after, mom_before), "Momentum should update even when Magma masks the update"
            break

    assert found_masked_step, "Expected at least one masked step with survival_prob=0.5"


def test_setup_optimizer_supports_muon_magma_entry():
    config = GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=32)
    model = GPT(config)
    optimizer = model.setup_optimizer(
        matrix_optimizer="muon_magma",
        magma_survival_prob=0.4,
        magma_temperature=1.5,
        magma_ema_decay=0.8,
    )

    muon_groups = [g for g in optimizer.param_groups if g.get("kind") == "muon"]
    assert muon_groups, "Expected Muon groups for matrix parameters"
    for group in muon_groups:
        assert group.get("magma") is True
        assert group.get("survival_prob") == 0.4
        assert group.get("temperature") == 1.5
        assert group.get("ema_decay") == 0.8


def test_muon_magma_uses_regular_weight_decay():
    p_base = torch.tensor([[1.0, -2.0, 3.0], [-0.5, 0.25, -1.5]], dtype=torch.float32)
    grad = torch.tensor([[0.6, 0.2, 0.8], [0.1, 0.4, 0.5]], dtype=torch.float32)

    p_no_wd = torch.nn.Parameter(p_base.clone())
    p_wd = torch.nn.Parameter(p_base.clone())
    common = dict(
        kind="muon",
        lr=1e-2,
        momentum=0.95,
        ns_steps=2,
        beta2=0.95,
        magma=True,
        survival_prob=1.0,
        temperature=2.0,
        ema_decay=0.9,
    )
    opt_no_wd = MuonAdamW([{**common, "params": [p_no_wd], "weight_decay": 0.0}])
    wd = 0.1
    opt_wd = MuonAdamW([{**common, "params": [p_wd], "weight_decay": wd}])

    p_no_wd.grad = grad.clone()
    p_wd.grad = grad.clone()
    p_before = p_base.clone()
    opt_no_wd.step()
    opt_wd.step()

    scaled_lr = common["lr"] * max(1.0, p_base.shape[-2] / p_base.shape[-1]) ** 0.5
    expected_with_wd = p_no_wd.detach() - scaled_lr * wd * p_before
    assert torch.allclose(p_wd.detach(), expected_with_wd, atol=1e-7, rtol=1e-6)
