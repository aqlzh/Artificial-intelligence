import torch


class line(torch.autograd.Function):

    @staticmethod
    # 前向运算
    def forward(ctx, w, x, b):
        # y = w*x + b
        ctx.save_for_backward(w, x, b)
        return w * x + b

    @staticmethod
    # 反向传播
    def backward(ctx, grad_out):  # 上下文管理器，上一级的梯度
        w, x, b = ctx.saved_tensors
        grad_w = grad_out * x
        grad_x = grad_out * w
        grad_b = grad_out * 1
        return grad_w, grad_x, grad_b


w = torch.rand(2, 2, requires_grad=True)
x = torch.rand(2, 2, requires_grad=True)
b = torch.rand(2, 2, requires_grad=True)

out = line.apply(w, x, b)
out.backward(torch.ones(2, 2))

print(w, x, b)
print(w.grad, x.grad, b.grad)
