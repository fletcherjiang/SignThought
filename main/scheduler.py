import abc

class BaseLearningRateScheduler(abc.ABC):

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def state_dict(self):
        pass

    @abc.abstractmethod
    def load_state_dict(self, state_dict):
        pass


class transformer_learning_rate_scheduler(BaseLearningRateScheduler):
    def __init__(self, optimizer, dim_model, warmup_steps, K):
        # Model Optimizer
        self.optimizer = optimizer

        # Model Step
        self.model_step = -1

        # Scheduler Params
        self.dim_model = dim_model
        self.warmup_steps = warmup_steps
        self.K = K

    def step(self, ckpt_score=None):
        # Update Model Step
        self.model_step += 1
        s = self.model_step + 1

        # Update LR
        arg1 = s ** -0.5
        arg2 = s * (self.warmup_steps ** -1.5)
        self.optimizer.param_groups[0]['lr'] = self.K * self.dim_model ** -0.5 * min(arg1, arg2)

    def state_dict(self):
        return {'model_step': self.model_step,
                'dim_model': self.dim_model,
                'warmup_steps': self.warmup_steps,
                'K': self.K}

    def load_state_dict(self, state_dict):
        self.model_step = state_dict['model_step']
        self.dim_model = state_dict['dim_model']
        self.warmup_steps = state_dict['warmup_steps']
        self.K = state_dict['K']


class exponential_decay_transformer_learning_rate_scheduler:
    def __init__(self, optimizer, warmup_steps, lr_max, alpha, end_step):
        # Model Optimizer
        self.optimizer = optimizer

        # Model Step
        self.model_step = -1

        # Scheduler Params
        self.warmup_steps = warmup_steps
        self.lr_max = lr_max
        self.alpha = alpha
        self.end_step = end_step

    def step(self, ckpt_score=None):
        # Update Model Step
        self.model_step += 1
        s = self.model_step + 1

        # Update LR
        arg1 = s / self.warmup_steps * self.lr_max  # Warmup phase
        arg2 = self.lr_max * self.alpha ** (
                    (s - self.warmup_steps) / (self.end_step - self.warmup_steps))  # Decay phase
        self.optimizer.param_groups[0]['lr'] = min(arg1, arg2)

    def state_dict(self):
        return {'model_step': self.model_step,
                'warmup_steps': self.warmup_steps,
                'lr_max': self.lr_max,
                'alpha': self.alpha,
                'end_step': self.end_step}

    def load_state_dict(self, state_dict):
        self.model_step = state_dict['model_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.lr_max = state_dict['lr_max']
        self.alpha = state_dict['alpha']
        self.end_step = state_dict['end_step']