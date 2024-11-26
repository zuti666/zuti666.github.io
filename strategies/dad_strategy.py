 class DADStrategy(BaseStrategy):
    """Distribution Adaptive Distillation Strategy"""
    
    def __init__(self, model, optimizer, criterion,
                 teacher_model=None,
                 temperature: float = 2.0,
                 alpha: float = 0.5,
                 adaptation_rate: float = 0.1,
                 train_mb_size: int = 32,
                 eval_mb_size: int = 32,
                 device='cuda',
                 plugins=None,
                 evaluator=None):
                 
        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator
        )
        
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.adaptation_rate = adaptation_rate
        self.distribution_tracker = DistributionTracker()
        
    def _compute_distillation_loss(self, student_outputs: torch.Tensor,
                                 teacher_outputs: torch.Tensor,
                                 temperature: float) -> torch.Tensor:
        """计算蒸馏损失"""
        soft_targets = F.softmax(teacher_outputs / temperature, dim=1)
        student_log_softmax = F.log_softmax(student_outputs / temperature, dim=1)
        
        # KL散度作为蒸馏损失
        distillation_loss = F.kl_div(
            student_log_softmax,
            soft_targets,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        return distillation_loss
        
    def _compute_adaptation_rate(self) -> float:
        """计算自适应率"""
        distribution_diff = self.distribution_tracker.compute_difference()
        return self.adaptation_rate * distribution_diff
        
    def training_epoch(self, **kwargs):
        """训练一个epoch"""
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break
                
            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)
            
            # 教师模型预测
            with torch.no_grad():
                teacher_outputs = self.teacher_model(self.mb_x)
            
            # 学生模型预测
            student_outputs = self.model(self.mb_x)
            
            # 计算损失
            task_loss = self.criterion(student_outputs, self.mb_y)
            distillation_loss = self._compute_distillation_loss(
                student_outputs,
                teacher_outputs,
                self.temperature
            )
            
            # 自适应调整alpha
            current_alpha = self.alpha * self._compute_adaptation_rate()
            
            # 总损失
            self.loss = current_alpha * distillation_loss + \
                       (1 - current_alpha) * task_loss
            
            # 更新模型
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
            
            self._after_training_iteration(**kwargs)