---
layout: wiki
title:  Python环境conda和pip的常用命令
categories: [Linux]
description: Conda PIP使用
keywords: [Linux, conda, pip ]
---


# Python环境conda和pip的常用命令

# conda

以下是一些 `conda` 常用的命令，涵盖环境管理和包管理的基础操作：

## 环境管理

1. **创建新环境**：
   
   ```bash
   conda create --name myenv
   ```
   创建一个名为 `myenv` 的新环境。
   
2. **激活环境**：
   ```bash
   conda activate myenv
   ```
   激活名为 `myenv` 的环境。

3. **停用当前环境**：
   ```bash
   conda deactivate
   ```

4. **列出所有环境**：
   ```bash
   conda env list
   ```
   或者：
   ```bash
   conda info --envs
   ```

5. **删除环境**：
   ```bash
   conda remove --name myenv --all
   ```
   删除名为 `myenv` 的环境。

6. **克隆环境**：
   ```bash
   conda create --name newenv --clone myenv
   ```
   克隆现有的环境 `myenv` 到 `newenv`。

7. **导出环境**：
   ```bash
   conda env export > environment.yml
   ```
   导出当前环境的所有包信息到 `environment.yml` 文件。

8. **从文件导入环境**：
   
   ```bash
   conda env create -f environment.yml
   ```
   通过 `environment.yml` 文件重新创建环境。

## 包管理

1. **安装包**：
   
   ```bash
   conda install package_name
   ```
   安装指定的包。
   
2. **指定版本安装包**：
   
   ```bash
   conda install package_name=2.0
   ```
   安装指定版本的包。
   
3. **更新包**：
   ```bash
   conda update package_name
   ```

4. **删除包**：
   
   ```bash
   conda remove package_name
   ```
   
5. **列出当前环境中的所有包**：
   ```bash
   conda list
   ```

6. **更新 `conda` 自身**：
   
   ```bash
   conda update conda
   ```

## 其他有用命令

1. **搜索包**：
   
   ```bash
   conda search package_name
   ```
   
2. **查看当前环境的详细信息**：
   ```bash
   conda info
   ```

这些命令涵盖了日常使用 `conda` 管理环境和包的基本需求。

# pip

`PIP` 是 Python 的包管理工具，全称为 "Pip Installs Packages"。它允许用户从 Python Package Index (PyPI) 下载并安装 Python 包。`pip` 是 Python 开发中非常常用的工具之一，支持包的安装、卸载和更新。下面介绍 `pip` 的一些基本用法和命令。

## PIP 的基本命令

### 1. **安装包**

安装指定的 Python 包可以使用以下命令：

```bash
pip install package_name
```

例如，安装 `requests` 包：

```bash
pip install requests
```

### 2. **安装指定版本的包**

如果需要安装特定版本的包，可以在包名后面加上版本号：

```bash
pip install package_name==1.2.3
```

例如，安装 `requests` 的 2.25.1 版本：

```bash
pip install requests==2.25.1
```

### 3. **更新包**

更新一个已安装的包到最新版本：

```bash
pip install --upgrade package_name
```

例如，更新 `requests` 包：

```bash
pip install --upgrade requests
```

### 4. **卸载包**

要卸载某个包，使用以下命令：

```bash
pip uninstall package_name
```

例如，卸载 `requests` 包：

```bash
pip uninstall requests
```

### 5. **列出已安装的包**

要查看当前环境中已安装的所有包及其版本，可以使用：

```bash
pip list
```

### 6. **查看包的详细信息**

可以使用 `pip show` 命令查看某个包的详细信息，例如包的安装位置、版本、依赖项等：

```bash
pip show package_name
```

例如，查看 `requests` 包的信息：

```bash
pip show requests
```

### 7. **搜索包**

可以使用 `pip search` 搜索 PyPI 中的包：

```bash
pip search package_name
```

例如，搜索与 `requests` 相关的包：

```bash
pip search requests
```

### 8. **冻结已安装的包**

`pip freeze` 命令可以输出所有已安装包及其版本号，通常用于生成一个 `requirements.txt` 文件，方便环境的重现：

```bash
pip freeze > requirements.txt
```

### 9. **从 `requirements.txt` 文件安装包**

如果你有一个包含包信息的 `requirements.txt` 文件，可以通过以下命令安装文件中列出的所有包：

```bash
pip install -r requirements.txt
```

### 10. **检查过时的包**

可以使用 `pip list --outdated` 来查看当前环境中有哪些包有新版本可以更新：

```bash
pip list --outdated
```

## PIP 的常用选项

- `--no-cache-dir`：不使用缓存来安装包，确保安装最新的包版本。
  
  ```bash
  pip install --no-cache-dir package_name
  ```

- `--user`：在当前用户的目录中安装包，而不是全局安装。通常用于没有管理员权限的情况。

  ```bash
  pip install --user package_name
  ```

## PIP 的高级功能

1. **使用镜像源**：
   在某些网络条件下，可能需要使用国内的 PyPI 镜像源来加快安装速度。可以通过以下方式指定镜像源：

   ```bash
   pip install package_name -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. **安装本地包**：
   如果你有一个本地的 `.whl` 或 `.tar.gz` 文件，可以通过 `pip` 安装：

   ```bash
   pip install /path/to/your/package.whl
   ```



`PIP` 是 Python 包管理的重要工具，通过它可以轻松管理 Python 项目的依赖包。无论是安装、卸载、更新，还是生成和使用 `requirements.txt` 文件，`pip` 提供了强大的功能来管理包的生命周期。
