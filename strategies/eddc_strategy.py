import torch
import numpy as np
from typing import Optional, Sequence
from avalanche.training import BaseStrategy
from avalanche.training.plugins import SupervisedPlugin
from torch.nn import functional as F

class EmbeddingDriftDetector:
    """嵌入漂移检测器"""
    def __init__(self, feature_dim: int = 512, window_size: int = 1000):
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.feature_history = []
        self.covariance_history = []
        
    def update_history(self, features: torch.Tensor):
        """更新特征历史"""
        features = features.detach().cpu().numpy()
        self.feature_history.extend(features)
        if len(self.feature_history) > self.window_size:
            self.feature_history = self.feature_history[-self.window_size:]
            
        # 更新协方差矩阵
        if len(self.feature_history) >= 100:  # 最小样本数
            cov = np.cov(np.array(self.feature_history).T)
            self.covariance_history.append(cov)
    
    def detect(self, current_features: torch.Tensor, 
              reference_features: Optional[torch.Tensor] = None) -> float:
        """检测特征漂移"""
        self.update_history(current_features)
        
        if len(self.covariance_history) < 2:
            return 0.0
            
        # 计算协方差矩阵的变化
        current_cov = self.covariance_history[-1]
        previous_cov = self.covariance_history[-2]
        
        # 使用Frobenius范数计算差异
        drift_score = np.linalg.norm(current_cov - previous_cov, ord='fro')
        
        # 归一化drift score到[0,1]区间
        drift_score = min(1.0, drift_score / (np.linalg.norm(previous_cov, ord='fro') + 1e-6))
        
        return drift_score

class DriftCorrectionModule:
    """漂移修正模块"""
    def __init__(self, model: torch.nn.Module, correction_rate: float = 0.1):
        self.model = model
        self.correction_rate = correction_rate
        self.feature_bank = []
        self.label_bank = []
        
    def store_features(self, features: torch.Tensor, labels: torch.Tensor):
        """存储特征和标签"""
        self.feature_bank.append(features.detach())
        self.label_bank.append(labels.detach())
        
        # 限制存储大小
        if len(self.feature_bank) > 100:
            self.feature_bank.pop(0)
            self.label_bank.pop(0)
    
    def compute_correction(self, current_features: torch.Tensor, 
                         drift_score: float) -> torch.Tensor:
        """计算特征修正"""
        if not self.feature_bank:
            return current_features
            
        # 获取历史特征中心
        historical_features = torch.cat(self.feature_bank, dim=0)
        feature_center = historical_features.mean(dim=0, keepdim=True)
        
        # 基于漂移分数计算修正强度
        correction_strength = self.correction_rate * drift_score
        
        # 应用修正
        corrected_features = (1 - correction_strength) * current_features + \
                           correction_strength * feature_center
        
        return corrected_features

class EDDCStrategy(BaseStrategy):
    """Embedding Drift Detection and Correction Strategy"""
    
    def __init__(self, model, optimizer, criterion,
                 drift_threshold: float = 0.5,
                 correction_rate: float = 0.1,
                 feature_dim: int = 512,
                 train_mb_size: int = 32,
                 eval_mb_size: int = 32,
                 device='cuda',
                 plugins: Optional[Sequence[SupervisedPlugin]] = None,
                 evaluator=None):
        
        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator
        )
        
        self.drift_threshold = drift_threshold
        self.drift_detector = EmbeddingDriftDetector(feature_dim=feature_dim)
        self.correction_module = DriftCorrectionModule(
            model, correction_rate=correction_rate
        )
        self.current_drift_score = 0.0
        
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        # 假设模型的倒数第二层是特征层
        features = None
        def hook(model, input, output):
            nonlocal features
            features = output.detach()
            
        handle = list(self.model.children())[-2].register_forward_hook(hook)
        self.model(x)
        handle.remove()
        
        return features
        
    def _apply_drift_correction(self, features: torch.Tensor, 
                              drift_score: float) -> torch.Tensor:
        """应用漂移修正"""
        return self.correction_module.compute_correction(features, drift_score)
        
    def training_epoch(self, **kwargs):
        """训练一个epoch"""
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break
                
            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            # 提取特征
            features = self._extract_features(self.mb_x)
            
            # 检测漂移
            self.current_drift_score = self.drift_detector.detect(features)
            
            # 如果检测到显著漂移，应用修正
            if self.current_drift_score > self.drift_threshold:
                features = self._apply_drift_correction(
                    features, self.current_drift_score
                )
                
            # 存储当前特征用于未来参考
            self.correction_module.store_features(features, self.mb_y)
            
            # 正常的训练步骤
            self.optimizer.zero_grad()
            self.loss = self._forward()
            self._backward()
            self.optimizer.step()
            
            self._after_training_iteration(**kwargs) 