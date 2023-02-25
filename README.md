<!--
 * @Author: WANG Maonan
 * @Date: 2023-02-25 18:39:16
 * @Description: 仓库说明
 * @LastEditTime: 2023-02-25 19:57:44
-->
# A universal TSC with delayed observations

## 文件介绍

New Start!

## Git 快速使用

每次开始写代码之前，首先更新仓库，同时创建一个新的分支：

```shell
git pull # 从主分支上更新
git checkout -b 2022_01_17_wmn # 创建新分支
```

接着在完成代码之后，进行上传：

```shell
git add . # 提交修改
git commit -m '修改了 xxx' # 对本次修改的说明
git push -u origin 2022_01_17_wmn # 将新的分支更新
git checkout master # 切换到主分支
```

提交之后，打开 Github 的仓库页面，点击「Compare & pull request」，接着点击「Create pull request」。
可以在「Files Change」看到修改的内容，确保没有问题之后，可以点击「Merge pull request」，即可将这个分支和主分支合并。
**这里一般不要自己合并，你写完我检查之后再合并。**

最后我们将本地的分支删除即可：

```shell
git branch -D 2022_01_17_wmn # 删除本地分支
```