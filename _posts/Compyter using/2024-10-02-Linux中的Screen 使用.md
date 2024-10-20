---
layout: post
title:  Screen 的使用
categories: [Linux ] 
description: Linux中的Screen 使用
keywords: [Liunx, Screen ] 
---



# Screen 的使用记录

为什么要使用 screen 

默认情况下，关闭 SSH 连接时，`screen` 中的所有进程会继续运行，这在长时间运行脚本或任务时非常有用。即使网络连接断开，也不必担心进程被终止。

## 常见选项总结：

- `screen -S [session_name]`：创建新会话并命名。
- `screen -ls`：查看所有活动的 `screen` 会话。
- `screen -r [session_name]`：重新连接到指定会话。
- `Ctrl + A，然后按 D`：分离当前会话。
- - 

## 关闭 screen 

### 退出关闭单个会话

可以通过以下操作完全退出 `screen` 会话：

- 在 `screen` 内输入 `exit` 命令。
- 或者直接通过 `Ctrl + D` 键组合



### 关闭多个screen 



#### 使用脚本来关闭前 3 个 



```bash
screen -ls | awk '/[0-9]+/{print $1}' | head -n 3 | xargs -I {} screen -X -S {} quit

```



#### 强制关闭 所有 screen 



```bash
pkill screen 
```

`pkill` 是 Linux/Unix 系统中的一个命令，用于根据进程名（或其他属性）终止进程。它的功能类似于 `kill`，但更简便，因为 `pkill` 不需要指定进程的 PID（进程ID），而是通过进程的名称或其他模式匹配来自动找到并终止目标进程。

这是一种非常快速关闭所有 `screen` 会话的方式，但与手动退出或使用 `screen -X -S <session_id> quit` 不同，它是强制性的。

#### 关闭特定的screen  

使用会话 ID 杀死某个特定会话：

```bash
screen -X -S <session_id> quit
```



例如，若要关闭 ID 为 12345 的会话：

```bash
screen -X -S 12345 quit
```

