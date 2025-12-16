# Git 操作指南

## 1. Git 基础概念

### 1.1 什么是 Git？
Git 是一个分布式版本控制系统（DVCS），用于跟踪文件的变化，协调多人协作开发。与集中式版本控制系统（如 SVN）不同，Git 允许每个开发者拥有完整的代码库副本。

### 1.2 Git 的主要特点
- **分布式架构**：每个开发者都有完整的代码库
- **高效的分支管理**：创建、合并分支操作快速
- **强大的撤销能力**：可以轻松回滚到任意历史版本
- **数据完整性**：使用 SHA-1 哈希确保数据不被篡改
- **离线工作**：大多数操作不需要网络连接

## 2. Git 初始化操作

### 2.1 安装 Git
- **Windows**：从 [Git 官网](https://git-scm.com/download/win) 下载安装包
- **macOS**：使用 Homebrew 安装：`brew install git`
- **Linux**：使用包管理器安装，如 `sudo apt-get install git`（Ubuntu）

### 2.2 配置 Git
设置用户信息（全局配置）：
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

查看配置信息：
```bash
git config --list
git config user.name
```

### 2.3 初始化仓库

#### 2.3.1 在现有目录中初始化
```bash
cd /path/to/your/project
git init
```
这会在当前目录创建一个 `.git` 子目录，包含 Git 仓库的所有必要文件。

#### 2.3.2 克隆远程仓库
```bash
git clone <repository-url>
# 示例：git clone https://github.com/user/repo.git

# 克隆到指定目录
git clone <repository-url> <directory-name>
```

## 3. Git 核心概念：工作区、暂存区、版本库

### 3.1 工作区（Working Directory）
你当前看到的目录，包含实际文件的地方。

### 3.2 暂存区（Stage/Index）
位于 `.git` 目录中的一个文件，用于临时存放要提交的文件快照，也称为索引。

### 3.3 版本库（Repository）
包含所有提交历史、分支、标签等信息的地方，位于 `.git` 目录中。

### 3.4 文件状态转换
- **未跟踪（Untracked）**：新创建的文件，Git 尚未开始跟踪
- **已修改（Modified）**：已跟踪的文件发生了变化，但尚未添加到暂存区
- **已暂存（Staged）**：已修改的文件已添加到暂存区，准备提交
- **已提交（Committed）**：文件已被安全地保存在版本库中

## 4. 文件状态管理

### 4.1 查看文件状态
```bash
git status
# 简洁输出
git status -s
```

### 4.2 添加文件到暂存区
```bash
# 添加单个文件
git add filename.txt

# 添加多个文件
git add file1.txt file2.txt

# 添加目录及所有内容
git add directory/

# 添加所有修改的文件
git add .
```

### 4.3 提交更改
```bash
git commit -m "Commit message"

# 跳过暂存区，直接提交所有修改的文件（仅跟踪过的文件）
git commit -a -m "Commit message"

# 修改最后一次提交（谨慎使用，特别是已推送到远程的提交）
git commit --amend -m "New commit message"
```

### 4.4 查看更改
```bash
# 查看工作区与暂存区的差异
git diff

# 查看暂存区与版本库的差异
git diff --staged

# 查看工作区与版本库的差异
git diff HEAD

# 查看两个版本之间的差异
git diff commit1 commit2
```

### 4.5 移除文件
```bash
# 从工作区和暂存区同时删除
git rm filename.txt

# 仅从暂存区删除，保留工作区文件
git rm --cached filename.txt
```

### 4.6 重命名/移动文件
```bash
git mv oldname.txt newname.txt
# 等价于：
# mv oldname.txt newname.txt
# git rm oldname.txt
# git add newname.txt
```

## 5. 分支管理

### 5.1 查看分支
```bash
# 查看本地分支
git branch

# 查看远程分支
git branch -r

# 查看所有分支（本地+远程）
git branch -a

# 查看分支及其最后一次提交
git branch -v
```

### 5.2 创建分支
```bash
# 创建新分支
git branch new-branch

# 创建并切换到新分支
git checkout -b new-branch
# Git 2.23+ 推荐使用：
git switch -c new-branch
```

### 5.3 切换分支
```bash
git checkout existing-branch
# Git 2.23+ 推荐使用：
git switch existing-branch

# 切换到上一个分支
git checkout -
git switch -
```

### 5.4 合并分支
```bash
# 切换到目标分支
git checkout main
# 合并 feature 分支到 main 分支
git merge feature-branch
```

#### 5.4.1 合并类型
- **快进合并（Fast-forward）**：当目标分支是当前分支的直接祖先时
- **三方合并（Three-way merge）**：创建一个新的合并提交，包含两个分支的更改
- ** squash 合并**：将多个提交压缩为一个提交再合并
  ```bash
  git merge --squash feature-branch
  git commit -m "Squash merge feature-branch"
  ```

### 5.5 删除分支
```bash
# 删除本地分支（确保已合并）
git branch -d branch-name

# 强制删除本地分支（即使未合并）
git branch -D branch-name

# 删除远程分支
git push origin --delete remote-branch-name
```

### 5.6 解决合并冲突
当两个分支修改了同一文件的相同部分时，会产生合并冲突。Git 会在冲突文件中标记冲突部分，需要手动解决：

1. 查看冲突文件：`git status`
2. 编辑文件，解决冲突标记
3. 将解决后的文件添加到暂存区：`git add filename.txt`
4. 提交合并结果：`git commit -m "Resolve merge conflict"`

## 6. 远程仓库操作

### 6.1 查看远程仓库
```bash
# 查看配置的远程仓库
git remote

# 查看远程仓库详细信息
git remote -v

# 查看特定远程仓库信息
git remote show origin
```

### 6.2 添加远程仓库
```bash
git remote add origin <repository-url>
```

### 6.3 重命名远程仓库
```bash
git remote rename old-name new-name
```

### 6.4 删除远程仓库
```bash
git remote remove remote-name
```

### 6.5 推送更改
```bash
# 推送本地分支到远程仓库
git push origin local-branch

# 推送并设置上游分支
git push -u origin local-branch

# 推送所有分支
git push --all origin

# 推送标签
git push --tags
```

### 6.6 获取远程更改

#### 6.6.1 git fetch
只下载远程更新，不合并：
```bash
# 获取所有远程分支更新
git fetch origin

# 获取特定分支更新
git fetch origin branch-name
```

#### 6.6.2 git pull
下载并合并远程更新（相当于 git fetch + git merge）：
```bash
# 拉取并合并当前分支的远程更新
git pull

# 拉取特定分支并合并到当前分支
git pull origin branch-name

# 使用 rebase 方式合并
git pull --rebase origin branch-name
```

## 7. 撤销操作

### 7.1 撤销工作区的修改
```bash
# 撤销单个文件的修改
git checkout -- filename.txt
# Git 2.23+ 推荐使用：
git restore filename.txt

# 撤销所有文件的修改
git checkout -- .
git restore .
```

### 7.2 撤销暂存区的修改
```bash
# 将文件从暂存区移除，但保留工作区修改
git reset HEAD filename.txt
# Git 2.23+ 推荐使用：
git restore --staged filename.txt

# 撤销所有暂存区修改
git reset HEAD
git restore --staged .
```

### 7.3 回退版本

#### 7.3.1 查看提交历史
```bash
git log
git log --oneline  # 简洁输出
git log --graph  # 图形化展示
git log --pretty=format:"%h %ad %s [%an]" --date=short  # 自定义格式
```

#### 7.3.2 回退到指定版本
```bash
# 回退到上一个版本
git reset --hard HEAD~1

# 回退到指定提交
git reset --hard <commit-hash>

# 保留工作区修改，只回退版本库和暂存区
git reset --mixed <commit-hash>

# 只回退版本库，保留暂存区和工作区
git reset --soft <commit-hash>
```

#### 7.3.3 找回丢失的提交
如果不小心回退了错误的版本，可以使用 `git reflog` 查看所有操作历史，然后通过 `git reset` 恢复：
```bash
git reflog
git reset --hard <commit-hash>
```

### 7.4 撤销提交
```bash
# 撤销最后一次提交，但保留更改在工作区
git reset --soft HEAD~1

# 撤销最后一次提交，并丢弃更改
git reset --hard HEAD~1

# 撤销特定提交（创建新的提交来抵消旧提交的更改）
git revert <commit-hash>
```

## 8. 标签管理

标签用于标记重要的提交点，如版本发布。

### 8.1 查看标签
```bash
git tag
git tag -l "v1.*"  # 查看特定模式的标签
```

### 8.2 创建标签

#### 8.2.1 轻量级标签
```bash
git tag v1.0.0
```

#### 8.2.2 带注释的标签
```bash
git tag -a v1.0.0 -m "Version 1.0.0 release"
```

#### 8.2.3 为旧提交创建标签
```bash
git tag -a v0.9.0 <commit-hash>
```

### 8.3 推送标签
```bash
# 推送特定标签
git push origin v1.0.0

# 推送所有标签
git push origin --tags
```

### 8.4 删除标签
```bash
# 删除本地标签
git tag -d v1.0.0

# 删除远程标签
git push origin --delete v1.0.0
```

## 9. 日志与历史查看

### 9.1 查看提交历史
```bash
git log
git log --oneline
git log --graph --oneline --decorate
```

### 9.2 查看文件历史
```bash
git log -- filename.txt
git log -p filename.txt  # 查看文件内容变化
git log --stat filename.txt  # 查看文件修改统计
```

### 9.3 查看提交差异
```bash
git diff commit1 commit2
git diff commit1 commit2 -- filename.txt  # 查看特定文件差异
```

### 9.4 查看分支图
```bash
git log --graph --oneline --all
# 更直观的分支图（需要安装 gitk）
gitk --all
```

## 10. 其他常用操作

### 10.1 忽略文件
创建 `.gitignore` 文件，列出不需要跟踪的文件和目录：
```
# 忽略所有 .log 文件
*.log

# 忽略 node_modules 目录
node_modules/

# 忽略特定文件
.env
.DS_Store
```

### 10.2 查看 Git 仓库大小
```bash
du -sh .git/
```

### 10.3 清理未跟踪文件
```bash
# 查看将被清理的文件
git clean -n

# 清理所有未跟踪文件
git clean -f

# 清理未跟踪文件和目录
git clean -fd
```

### 10.4 子模块管理
子模块允许将一个 Git 仓库作为另一个 Git 仓库的子目录。

```bash
# 添加子模块
git submodule add <repository-url> <path>

# 克隆包含子模块的仓库
git clone <repository-url>
git submodule init
git submodule update
# 或一步完成：git clone --recurse-submodules <repository-url>

# 更新子模块
git submodule update --remote
```

### 10.5 储藏（Stash）
暂时保存工作区和暂存区的更改，以便切换到其他分支：
```bash
# 储藏当前更改
git stash
git stash save "Stash message"

# 查看储藏列表
git stash list

# 应用最近的储藏
git stash apply

# 应用并删除储藏
git stash pop

# 应用特定储藏
git stash apply stash@{2}

# 删除储藏
git stash drop stash@{2}
git stash clear  # 删除所有储藏
```

### 10.6 交互式变基
用于重写提交历史，如合并、拆分、重命名提交：
```bash
git rebase -i HEAD~5  # 重写最近 5 个提交
```

## 11. Git 工作流程

### 11.1 常见工作流程
- **Git Flow**：适合大型项目，包含 master、develop、feature、release、hotfix 等分支
- **GitHub Flow**：适合持续部署，只有 master 分支和 feature 分支
- **GitLab Flow**：结合了 Git Flow 和 GitHub Flow 的特点

### 11.2 团队协作建议
- 定期拉取最新代码：`git pull`
- 频繁提交小的更改，而不是大的提交
- 写清晰的提交信息
- 使用分支进行新功能开发
- 在合并前进行代码审查

## 12. 常见问题与解决方案

### 12.1 撤销错误的合并
```bash
git reset --hard HEAD~1  # 如果刚合并
# 或使用 git revert
```

### 12.2 处理断网情况
Git 支持离线工作，所有更改都保存在本地，网络恢复后再推送。

### 12.3 解决权限问题
确保对远程仓库有正确的访问权限，通常使用 SSH 密钥或 HTTPS 凭证。

## 13. 学习资源

- [Pro Git 电子书](https://git-scm.com/book/en/v2)
- [Git 官方文档](https://git-scm.com/doc)
- [GitHub Git 教程](https://guides.github.com/introduction/git-handbook/)
- [Git 可视化工具](https://learngitbranching.js.org/)