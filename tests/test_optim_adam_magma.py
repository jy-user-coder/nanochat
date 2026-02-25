import torch

from nanochat.optim_adam_magma import AdamMagma


def test_adam_magma_updates_moments_on_masked_steps():
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32))
    opt = AdamMagma([{"params": [p], "lr": 1e-2, "magma": True, "survival_prob": 0.5}], betas=(0.9, 0.99))

    torch.manual_seed(1234)
    found_masked_step = False
    for _ in range(64):
        p.grad = torch.tensor([0.25, -0.5, 0.75], dtype=torch.float32)
        p_before = p.detach().clone()
        exp_avg_before = opt.state[p].get("exp_avg", torch.zeros_like(p)).detach().clone()
        opt.step()
        exp_avg_after = opt.state[p]["exp_avg"]

        if torch.allclose(p, p_before):
            found_masked_step = True
            assert not torch.allclose(exp_avg_after, exp_avg_before), "Moments should update even when update is masked"
            break

    assert found_masked_step, "Expected at least one masked step with survival_prob=0.5"


def test_adam_magma_matches_adamw_when_disabled():
    torch.manual_seed(0)
    p1 = torch.nn.Parameter(torch.randn(8, dtype=torch.float32))
    p2 = torch.nn.Parameter(p1.detach().clone())

    kwargs = dict(lr=3e-3, betas=(0.8, 0.95), eps=1e-9, weight_decay=0.01)
    opt_magma = AdamMagma([{"params": [p1], "magma": False}], **kwargs)
    opt_adamw = torch.optim.AdamW([p2], **kwargs)

    for _ in range(10):
        grad = torch.randn_like(p1)
        p1.grad = grad.clone()
        p2.grad = grad.clone()
        opt_magma.step()
        opt_adamw.step()

    assert torch.allclose(p1, p2, atol=1e-7, rtol=1e-6)


def test_adam_magma_uses_regular_decoupled_weight_decay():
    p_base = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
    grad = torch.tensor([0.25, -0.5, 0.75], dtype=torch.float32)
    lr = 3e-3
    wd = 0.1

    p_no_wd = torch.nn.Parameter(p_base.clone())
    p_wd = torch.nn.Parameter(p_base.clone())
    common = dict(lr=lr, betas=(0.9, 0.99), eps=1e-8)
    opt_no_wd = AdamMagma([{"params": [p_no_wd], "magma": True, "survival_prob": 1.0}], weight_decay=0.0, **common)
    opt_wd = AdamMagma([{"params": [p_wd], "magma": True, "survival_prob": 1.0}], weight_decay=wd, **common)

    p_no_wd.grad = grad.clone()
    p_wd.grad = grad.clone()
    opt_no_wd.step()
    opt_wd.step()

    expected_with_wd = p_no_wd.detach() - lr * wd * p_base
    assert torch.allclose(p_wd.detach(), expected_with_wd, atol=1e-7, rtol=1e-6)
