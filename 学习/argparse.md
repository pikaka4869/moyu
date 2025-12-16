# Python argparse 模块完全指南

## 1. argparse 简介

argparse 是 Python 标准库中用于解析命令行参数的模块，它提供了一种简单而灵活的方式来编写用户友好的命令行界面。argparse 可以自动生成帮助和使用信息，并在用户提供无效参数时给出错误提示。

### 1.1 argparse 的主要特点

- **自动生成帮助和使用信息**：用户可以通过 `--help` 或 `-h` 参数查看详细的帮助信息
- **支持位置参数和可选参数**：灵活定义不同类型的命令行参数
- **参数类型检查**：可以指定参数的类型（如整数、浮点数、字符串等）
- **默认值设置**：可以为参数设置默认值
- **互斥参数支持**：可以定义互斥的参数组
- **子命令支持**：支持创建具有多个子命令的复杂命令行工具
- **自定义错误处理**：可以自定义参数验证和错误处理

### 1.2 argparse 的历史

argparse 模块是在 Python 3.2 版本中引入的，它替代了之前的 `optparse` 模块，提供了更强大和更灵活的功能。argparse 基于 `optparse` 和 `getopt` 模块的功能，但提供了更简洁的 API 和更多的功能。

## 2. argparse 基础用法

argparse 的基本用法包括三个步骤：

1. 创建解析器对象
2. 添加参数
3. 解析参数

### 2.1 创建解析器对象

首先，需要创建一个 `ArgumentParser` 对象，它负责解析命令行参数：

```python
import argparse

parser = argparse.ArgumentParser(description='这是一个简单的命令行工具')
```

`ArgumentParser` 构造函数接受多个参数来配置解析器：

- `description`：工具的简短描述，会显示在帮助信息中
- `prog`：程序的名称（默认使用 `sys.argv[0]`）
- `usage`：自定义使用信息
- `epilog`：帮助信息末尾的额外文本

### 2.2 添加参数

使用 `add_argument()` 方法向解析器添加参数：

```python
parser.add_argument('echo', help='要回显的字符串')
```

### 2.3 解析参数

使用 `parse_args()` 方法解析命令行参数：

```python
args = parser.parse_args()
print(args.echo)
```

完整的示例：

```python
import argparse

# 创建解析器
parser = argparse.ArgumentParser(description='这是一个简单的回显程序')

# 添加参数
parser.add_argument('echo', help='要回显的字符串')

# 解析参数
args = parser.parse_args()

# 使用参数
print(args.echo)
```

运行这个脚本：

```bash
python script.py hello
```

输出：

```
hello
```

## 3. 参数类型

argparse 支持两种主要类型的参数：位置参数和可选参数。

### 3.1 位置参数

位置参数是必须按顺序提供的参数，类似于函数的位置参数：

```python
import argparse

parser = argparse.ArgumentParser(description='计算两个数的和')
parser.add_argument('x', type=int, help='第一个数')
parser.add_argument('y', type=int, help='第二个数')

args = parser.parse_args()
print(f'{args.x} + {args.y} = {args.x + args.y}')
```

运行：

```bash
python script.py 5 3
```

输出：

```
5 + 3 = 8
```

### 3.2 可选参数

可选参数以 `-` 或 `--` 开头，用户可以选择是否提供：

```python
import argparse

parser = argparse.ArgumentParser(description='计算两个数的和')
parser.add_argument('--x', type=int, help='第一个数')
parser.add_argument('--y', type=int, help='第二个数')

args = parser.parse_args()
print(f'{args.x} + {args.y} = {args.x + args.y}')
```

运行：

```bash
python script.py --x 5 --y 3
```

输出：

```
5 + 3 = 8
```

### 3.3 短选项

可选参数可以同时有长选项和短选项：

```python
import argparse

parser = argparse.ArgumentParser(description='计算两个数的和')
parser.add_argument('-x', '--x_value', type=int, help='第一个数')
parser.add_argument('-y', '--y_value', type=int, help='第二个数')

args = parser.parse_args()
print(f'{args.x_value} + {args.y_value} = {args.x_value + args.y_value}')
```

运行：

```bash
python script.py -x 5 -y 3
```

输出：

```
5 + 3 = 8
```

### 3.4 必选可选参数

默认情况下，可选参数是可选的，但可以通过 `required=True` 使其成为必选：

```python
import argparse

parser = argparse.ArgumentParser(description='计算两个数的和')
parser.add_argument('--x', type=int, required=True, help='第一个数（必选）')
parser.add_argument('--y', type=int, required=True, help='第二个数（必选）')

args = parser.parse_args()
print(f'{args.x} + {args.y} = {args.x + args.y}')
```

如果不提供这些参数，将显示错误：

```bash
python script.py
```

输出：

```
usage: script.py [-h] --x X --y Y
script.py: error: the following arguments are required: --x, --y
```

## 4. 参数配置选项

argparse 提供了丰富的选项来配置参数的行为：

### 4.1 参数类型

可以通过 `type` 参数指定参数的类型：

```python
import argparse

parser = argparse.ArgumentParser(description='参数类型示例')
parser.add_argument('--int', type=int, help='整数参数')
parser.add_argument('--float', type=float, help='浮点数参数')
parser.add_argument('--str', type=str, help='字符串参数')
parser.add_argument('--file', type=open, help='文件参数')

args = parser.parse_args()
```

### 4.2 默认值

可以通过 `default` 参数为参数设置默认值：

```python
import argparse

parser = argparse.ArgumentParser(description='默认值示例')
parser.add_argument('--name', type=str, default='World', help='名称')
parser.add_argument('--times', type=int, default=1, help='次数')

args = parser.parse_args()
for _ in range(args.times):
    print(f'Hello, {args.name}!')
```

运行：

```bash
python script.py
```

输出：

```
Hello, World!
```

运行：

```bash
python script.py --name Alice --times 3
```

输出：

```
Hello, Alice!
Hello, Alice!
Hello, Alice!
```

### 4.3 帮助信息

可以通过 `help` 参数为参数提供帮助信息：

```python
import argparse

parser = argparse.ArgumentParser(description='帮助信息示例')
parser.add_argument('--name', type=str, help='用户名称')
parser.add_argument('--age', type=int, help='用户年龄')

args = parser.parse_args()
```

运行 `python script.py --help` 查看帮助信息：

```
usage: script.py [-h] [--name NAME] [--age AGE]

帮助信息示例

optional arguments:
  -h, --help     show this help message and exit
  --name NAME    用户名称
  --age AGE      用户年龄
```

### 4.4 选择范围

可以通过 `choices` 参数限制参数的取值范围：

```python
import argparse

parser = argparse.ArgumentParser(description='选择范围示例')
parser.add_argument('--color', choices=['red', 'green', 'blue'], help='颜色选择')
parser.add_argument('--number', type=int, choices=range(1, 6), help='数字选择')

args = parser.parse_args()
print(f'颜色: {args.color}')
print(f'数字: {args.number}')
```

如果提供的参数不在选择范围内，将显示错误：

```bash
python script.py --color yellow
```

输出：

```
usage: script.py [-h] [--color {red,green,blue}] [--number {1,2,3,4,5}]
script.py: error: argument --color: invalid choice: 'yellow' (choose from 'red', 'green', 'blue')
```

### 4.5 多值参数

可以通过 `nargs` 参数指定参数可以接受多个值：

```python
import argparse

parser = argparse.ArgumentParser(description='多值参数示例')
parser.add_argument('--files', nargs='+', help='多个文件路径')
parser.add_argument('--numbers', nargs=3, type=int, help='恰好三个数字')

args = parser.parse_args()
print(f'文件列表: {args.files}')
print(f'数字列表: {args.numbers}')
```

运行：

```bash
python script.py --files file1.txt file2.txt file3.txt --numbers 1 2 3
```

输出：

```
文件列表: ['file1.txt', 'file2.txt', 'file3.txt']
数字列表: [1, 2, 3]
```

`nargs` 参数的取值：

- `N`：恰好 N 个值
- `?`：0 或 1 个值
- `*`：0 或多个值
- `+`：1 或多个值

### 4.6 布尔值参数

可以通过 `action='store_true'` 或 `action='store_false'` 创建布尔值参数：

```python
import argparse

parser = argparse.ArgumentParser(description='布尔值参数示例')
parser.add_argument('--verbose', action='store_true', help='详细输出')
parser.add_argument('--quiet', action='store_false', help='静默模式')

args = parser.parse_args()

if args.verbose:
    print('详细模式已启用')
if args.quiet:
    print('静默模式已禁用')
```

运行：

```bash
python script.py --verbose
```

输出：

```
详细模式已启用
静默模式已禁用
```

### 4.7 计数参数

可以通过 `action='count'` 创建计数参数：

```python
import argparse

parser = argparse.ArgumentParser(description='计数参数示例')
parser.add_argument('-v', '--verbose', action='count', default=0, help='详细级别')

args = parser.parse_args()

if args.verbose == 0:
    print('正常输出')
elif args.verbose == 1:
    print('详细输出级别 1')
elif args.verbose >= 2:
    print('详细输出级别 2')
```

运行：

```bash
python script.py -vv
```

输出：

```
详细输出级别 2
```

## 5. argparse 高级功能

### 5.1 互斥参数

可以通过 `add_mutually_exclusive_group()` 创建互斥参数组：

```python
import argparse

parser = argparse.ArgumentParser(description='互斥参数示例')
group = parser.add_mutually_exclusive_group()
group.add_argument('--add', action='store_true', help='添加模式')
group.add_argument('--remove', action='store_true', help='删除模式')
group.add_argument('--update', action='store_true', help='更新模式')

args = parser.parse_args()

if args.add:
    print('添加模式')
elif args.remove:
    print('删除模式')
elif args.update:
    print('更新模式')
else:
    print('请选择模式')
```

如果同时提供互斥参数，将显示错误：

```bash
python script.py --add --remove
```

输出：

```
usage: script.py [-h] [--add | --remove | --update]
script.py: error: argument --remove: not allowed with argument --add
```

### 5.2 参数组

可以通过 `add_argument_group()` 创建参数组，使帮助信息更清晰：

```python
import argparse

parser = argparse.ArgumentParser(description='参数组示例')

# 创建参数组
input_group = parser.add_argument_group('输入参数')
output_group = parser.add_argument_group('输出参数')

# 添加参数到组
input_group.add_argument('--input', type=str, help='输入文件')
input_group.add_argument('--format', type=str, choices=['txt', 'csv', 'json'], help='输入格式')

output_group.add_argument('--output', type=str, help='输出文件')
output_group.add_argument('--verbose', action='store_true', help='详细输出')

args = parser.parse_args()
```

运行 `python script.py --help` 查看帮助信息：

```
usage: script.py [-h] [--input INPUT] [--format {txt,csv,json}] [--output OUTPUT] [--verbose]

参数组示例

optional arguments:
  -h, --help            show this help message and exit

输入参数:
  --input INPUT         输入文件
  --format {txt,csv,json}
                        输入格式

输出参数:
  --output OUTPUT       输出文件
  --verbose             详细输出
```

### 5.3 子命令

argparse 支持创建具有多个子命令的复杂命令行工具：

```python
import argparse

# 创建主解析器
parser = argparse.ArgumentParser(description='具有子命令的命令行工具')

# 创建子命令解析器
subparsers = parser.add_subparsers(dest='command', help='子命令')

# 创建 add 子命令
add_parser = subparsers.add_parser('add', help='添加两个数')
add_parser.add_argument('x', type=int, help='第一个数')
add_parser.add_argument('y', type=int, help='第二个数')

# 创建 multiply 子命令
multiply_parser = subparsers.add_parser('multiply', help='乘两个数')
multiply_parser.add_argument('x', type=int, help='第一个数')
multiply_parser.add_argument('y', type=int, help='第二个数')

args = parser.parse_args()

if args.command == 'add':
    print(f'{args.x} + {args.y} = {args.x + args.y}')
elif args.command == 'multiply':
    print(f'{args.x} * {args.y} = {args.x * args.y}')
else:
    parser.print_help()
```

运行：

```bash
python script.py add 5 3
```

输出：

```
5 + 3 = 8
```

运行：

```bash
python script.py multiply 5 3
```

输出：

```
5 * 3 = 15
```

### 5.4 自定义动作

可以通过 `action` 参数自定义参数的处理方式：

```python
import argparse

class SquareAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values * values)

parser = argparse.ArgumentParser(description='自定义动作示例')
parser.add_argument('--square', type=int, action=SquareAction, help='平方值')

args = parser.parse_args()
print(f'平方值: {args.square}')
```

运行：

```bash
python script.py --square 5
```

输出：

```
平方值: 25
```

## 6. argparse 用法示例

### 6.1 简单示例：计算器

```python
import argparse

# 创建解析器
parser = argparse.ArgumentParser(description='简单计算器')

# 添加参数
parser.add_argument('operation', type=str, choices=['add', 'subtract', 'multiply', 'divide'], help='运算类型')
parser.add_argument('x', type=float, help='第一个数')
parser.add_argument('y', type=float, help='第二个数')

# 解析参数
args = parser.parse_args()

# 执行运算
if args.operation == 'add':
    result = args.x + args.y
elif args.operation == 'subtract':
    result = args.x - args.y
elif args.operation == 'multiply':
    result = args.x * args.y
elif args.operation == 'divide':
    if args.y == 0:
        parser.error('除数不能为零')
    result = args.x / args.y

# 输出结果
print(f'{args.x} {args.operation} {args.y} = {result}')
```

运行：

```bash
python calculator.py add 5 3
python calculator.py divide 10 2
```

输出：

```
5.0 add 3.0 = 8.0
10.0 divide 2.0 = 5.0
```

### 6.2 示例：文件处理工具

```python
import argparse
import os

# 创建解析器
parser = argparse.ArgumentParser(description='文件处理工具')

# 创建子命令
subparsers = parser.add_subparsers(dest='command', help='子命令')

# 创建 list 子命令
list_parser = subparsers.add_parser('list', help='列出文件')
list_parser.add_argument('directory', type=str, default='.', nargs='?', help='目录路径')
list_parser.add_argument('--hidden', action='store_true', help='显示隐藏文件')

# 创建 delete 子命令
delete_parser = subparsers.add_parser('delete', help='删除文件')
delete_parser.add_argument('files', type=str, nargs='+', help='要删除的文件')
delete_parser.add_argument('--force', action='store_true', help='强制删除')

# 创建 rename 子命令
rename_parser = subparsers.add_parser('rename', help='重命名文件')
rename_parser.add_argument('old_name', type=str, help='旧文件名')
rename_parser.add_argument('new_name', type=str, help='新文件名')

# 解析参数
args = parser.parse_args()

# 执行命令
if args.command == 'list':
    # 列出文件
    files = os.listdir(args.directory)
    if not args.hidden:
        files = [f for f in files if not f.startswith('.')]
    for file in sorted(files):
        print(file)
elif args.command == 'delete':
    # 删除文件
    for file in args.files:
        try:
            os.remove(file)
            print(f'已删除: {file}')
        except Exception as e:
            if args.force:
                print(f'删除失败 (强制): {file}, 错误: {e}')
            else:
                parser.error(f'删除失败: {file}, 错误: {e}')
elif args.command == 'rename':
    # 重命名文件
    try:
        os.rename(args.old_name, args.new_name)
        print(f'已重命名: {args.old_name} -> {args.new_name}')
    except Exception as e:
        parser.error(f'重命名失败: {e}')
else:
    parser.print_help()
```

运行：

```bash
python file_tool.py list --hidden
python file_tool.py rename old.txt new.txt
python file_tool.py delete temp.txt --force
```

### 6.3 示例：数据处理脚本

```python
import argparse
import csv
import json

# 创建解析器
parser = argparse.ArgumentParser(description='数据处理脚本')

# 输入参数
parser.add_argument('--input', type=str, required=True, help='输入文件')
parser.add_argument('--input-format', type=str, choices=['csv', 'json'], required=True, help='输入格式')

# 输出参数
parser.add_argument('--output', type=str, required=True, help='输出文件')
parser.add_argument('--output-format', type=str, choices=['csv', 'json'], required=True, help='输出格式')

# 处理参数
parser.add_argument('--filter', type=str, help='过滤条件 (JSON 格式)')
parser.add_argument('--sort', type=str, help='排序字段')
parser.add_argument('--reverse', action='store_true', help='倒序排序')

# 解析参数
args = parser.parse_args()

# 加载数据
with open(args.input, 'r') as f:
    if args.input_format == 'csv':
        reader = csv.DictReader(f)
        data = list(reader)
    elif args.input_format == 'json':
        data = json.load(f)

# 过滤数据
if args.filter:
    filter_conditions = json.loads(args.filter)
    filtered_data = []
    for item in data:
        match = True
        for key, value in filter_conditions.items():
            if str(item.get(key, '')) != str(value):
                match = False
                break
        if match:
            filtered_data.append(item)
    data = filtered_data

# 排序数据
if args.sort:
    data.sort(key=lambda x: x.get(args.sort, ''), reverse=args.reverse)

# 保存数据
with open(args.output, 'w') as f:
    if args.output_format == 'csv':
        if not data:
            writer = csv.writer(f)
            writer.writerow([])
        else:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    elif args.output_format == 'json':
        json.dump(data, f, indent=2)

print(f'数据处理完成: {args.input} -> {args.output}')
print(f'处理记录数: {len(data)}')
```

运行：

```bash
python data_process.py --input data.csv --input-format csv --output data.json --output-format json --filter '{"status": "active"}' --sort name
```

## 7. argparse 最佳实践

### 7.1 命名约定

- 长选项使用下划线分隔的小写字母（如 `--file_path`）
- 短选项使用单个小写字母（如 `-f`）
- 参数名称使用下划线分隔的小写字母（如 `file_path`）

### 7.2 帮助信息

- 为每个参数提供清晰的帮助信息
- 使用 `description` 和 `epilog` 提供工具的概述和额外信息
- 使用参数组组织相关的参数

### 7.3 错误处理

- 使用 `parser.error()` 抛出自定义错误信息
- 为必填参数使用 `required=True` 而不是在解析后检查

### 7.4 测试

- 测试各种参数组合
- 测试无效参数的处理
- 测试帮助信息的显示

## 8. 常见问题

### 8.1 如何处理未知参数？

可以使用 `parse_known_args()` 方法处理未知参数：

```python
import argparse

parser = argparse.ArgumentParser(description='未知参数示例')
parser.add_argument('--known', type=str, help='已知参数')

args, unknown = parser.parse_known_args()

print(f'已知参数: {args.known}')
print(f'未知参数: {unknown}')
```

### 8.2 如何在脚本中使用默认值？

可以在 `ArgumentParser` 构造函数中使用 `add_argument()` 的 `default` 参数：

```python
import argparse

parser = argparse.ArgumentParser(description='默认值示例')
parser.add_argument('--name', type=str, default='World', help='名称')

args = parser.parse_args()
print(f'Hello, {args.name}!')
```

### 8.3 如何处理文件路径？

可以使用 `type=argparse.FileType('r')` 处理文件路径：

```python
import argparse

parser = argparse.ArgumentParser(description='文件处理示例')
parser.add_argument('--input', type=argparse.FileType('r'), help='输入文件')
parser.add_argument('--output', type=argparse.FileType('w'), help='输出文件')

args = parser.parse_args()

# 读取输入文件
content = args.input.read()

# 处理内容
processed_content = content.upper()

# 写入输出文件
args.output.write(processed_content)

# 关闭文件
args.input.close()
args.output.close()
```

## 9. 总结

argparse 是 Python 标准库中用于解析命令行参数的强大工具，它提供了丰富的功能来创建用户友好的命令行界面。通过学习 argparse 的基本用法和高级功能，您可以轻松地编写各种复杂的命令行工具。

argparse 的主要优势包括：

- 自动生成帮助和使用信息
- 支持位置参数和可选参数
- 提供参数类型检查和验证
- 支持默认值和选择范围
- 提供互斥参数和参数组
- 支持子命令

通过实践和练习，您将能够熟练地使用 argparse 来创建各种命令行工具，提高您的工作效率和代码质量。

## 10. 参考资源

- [Python 官方文档 - argparse](https://docs.python.org/3/library/argparse.html)
- [Python argparse Tutorial](https://docs.python.org/3/howto/argparse.html)
- [Real Python - Python argparse Tutorial](https://realpython.com/command-line-interfaces-python-argparse/)
- [Argparse Cookbook](https://docs.python.org/3/library/argparse.html#argparse-cookbook)