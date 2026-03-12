```
def _full_finetune(self, train_loader, test_loader, optimizer, scheduler, epochs):
        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                if self._optimizer_type == "cflat":
                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        logits = outputs["logits"]
                        loss = F.cross_entropy(logits, targets)
                        return outputs, [loss]

                    _, loss_list = optimizer.step(closure=closure)
                    loss_value = torch.stack([loss_term.detach() for loss_term in loss_list]).sum()
                    losses += loss_value.item()

                    with torch.no_grad():
                        logits = self._network(inputs)["logits"]
                elif self._optimizer_type == "gam":
                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        logits = outputs["logits"]
                        loss = F.cross_entropy(logits, targets)
                        loss_value = loss.detach()
                        loss.backward()  # ← 必须：让 p.grad 生效
                        return outputs, loss_value
                    outputs, loss_value = optimizer.step(closure=closure)
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    logits = outputs["logits"].detach()
                else:
                    optimizer.zero_grad()
                    logits = self._network(inputs)["logits"]
                    loss = F.cross_entropy(logits, targets)

                    if self._optimizer_type == "sam":
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        logits = self._network(inputs)["logits"]
                        second_loss = F.cross_entropy(logits, targets)
                        second_loss.backward()
                        optimizer.second_step(zero_grad=True)
                        losses += second_loss.item()
                    else:
                        loss.backward()
                        optimizer.step()
                        losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler is not None:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if (epoch+1) % 5 == 0 or epoch == epochs - 1:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "FullFinetune Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "FullFinetune Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)
```



T_max=self.args.get("init_epoch", None),



T_max=self.args.get("epochs", None)



self.args.get("full_epochs")







推荐的公平对比设定（两种预算，二选一）

- 预算 40（推荐，更接近你的 FT 基线）
  - FT：full_epochs=40，optimizer=SGD，momentum=0.9，weight_decay=0.002，scheduler=cosine（T_max=40），full_lr=0.01 或保持基线 0.005+MultiStep([15,25,35])
  - LPFT：LP=10ep（lr=0.015、wd=0.0005、常量 LR、backbone=eval），FT=30ep，其他与 FT 完全一致
  - LPFT‑EFM：同 LPFT，另加 EFM（LP：efm_lp_enable=true, efm_lambda_lp=0.005；FT：efm_ft_enable=true, efm_lambda_ft=0.03, efm_eta=0.001, efm_tau=1.2）
- 预算 30（更快的对照）
  - FT：full_epochs=30，SGD+cosine（T_max=30），full_lr=0.01，momentum=0.9，weight_decay=0.002
  - LPFT：LP=10ep（同上），FT=20ep（同 FT）
  - LPFT‑EFM：同 LPFT，EFM 同上
  - 备注：30ep 的 FAA 会比 40ep 略低，但三条线的差距仍可观察；更利于跑格。

统一的“经典协议”选择（便于被接受）

- 优化器：SGD（momentum=0.9）
- 调度：cosine（T_max=总 epoch），三条线一致
- 学习率/正则：full_lr=0.01（ViT 微调常见，配合 cosine 更好），weight_decay=0.002（你基线能学起来的值）
- Batch/增强/数据/seed：全部一致
- LP 阶段：8–10ep，lr=0.015、wd=0.0005，常量 LR，backbone=eval，fc-only
- 只让“是否 LP 与是否 EFM”成为方法差异

为什么这样更有说服力

- 避免对 baseline“精调到最优”而对新方法“欠调”的质疑：选取社区常用的 SGD+cosine/SGD+MultiStep 与通用 wd，保证三条线上一致，仅区别 LP 和 EFM。
- 预算一致、调度一致，曲线可对齐；LP 阶段短小，FT 为主，避免“LP 头被 FT 抹平”的假阴性。

落地改动（最小）

- 把三份 YAML 统一到上述协议（cosine 拼写在你当前代码是 “cosinelr” 可继续使用；wd 改为 0.002；统一 full_epochs；LP 头 10ep/0.015）。
- sgd_test5 的 FT 配置修正（保证能学起来）：
  - full_weight_decay: 0.002
  - scheduler: cosinelr（你当前支持）或 Multistep([10,20])（30ep）/([15,25,35])（40ep）
  - full_lr: 0.01（或基线 0.005+MultiStep）
- EFM 扫小格（仅 LPFT‑EFM）：efm_lambda_ft ∈ {0.03, 0.04}（0.03 起步更快，0.04 稍稳）

报告与指标

- 报 FAA（最终 Top‑1）+ 中期点（E15/E20/E30），可选 AAA（过程均值）以突出“稳态与中期更好”。
- 说明：采用统一经典协议（SGD+cosine），预算一致；LP 阶段 10ep 常量 LR 作为温和初始化；EFM 作为唯一正则差异，显著优于 LPFT 与 FT。

需要我直接把三份 YAML（FT/LPFT/LPFT‑EFM）按“预算 40：10/30”的协议补丁好并出一份 sweep 脚本（λ_ft ∈ {0.03,0.04}）吗？





lr = optimizer.param_groups[0]["lr"]





```
self._network.to(self._device)
        params = [p for p in self._network.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params, stage="full")
        lr = optimizer.param_groups[0]["lr"] 
        epochs = int(self.args.get("full_epochs"))
        scheduler = self.build_scheduler(
            optimizer,
            policy=self.args.get("scheduler", "constant"),
            T_max=epochs,
            milestones=self.args.get("full_milestones"),
            gamma=self.args.get("full_lr_decay"),
            eta_min=0.1*lr
        )
```

