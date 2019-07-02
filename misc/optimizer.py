import mxnet as mx
from mxnet import nd

@mx.optimizer.Optimizer.register
class RMSpropTorch(mx.optimizer.RMSProp):
    def __init__(self, **kwargs):
        super(RMSpropTorch, self).__init__(**kwargs)

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, nd.NDArray))
        assert(isinstance(grad, nd.NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)
        if self.clip_weights is not None:
            grad = clip(weight, -self.clip_weights, self.clip_weights)

        # only support not centered version
        assert not self.centered and self.gamma2 == 0
        (square_avg, ) = state

        grad[:] += wd * weight
        square_avg[:] *= self.gamma1
        square_avg[:] += (1 - self.gamma1) * grad * grad

        weight[:] += -lr * grad / (nd.sqrt(square_avg) + self.epsilon)
