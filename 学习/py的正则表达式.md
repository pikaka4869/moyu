# Python 正则表达式完全指南

## 1. 正则表达式基础概念

正则表达式（Regular Expression，简称 regex 或 regexp）是一种用于匹配字符串模式的强大工具。它使用一系列字符和特殊符号来定义一种搜索模式，用于文本查找、替换、验证和提取等操作。

### 1.1 正则表达式的用途

- **文本匹配**：验证字符串是否符合特定格式（如邮箱、URL、电话号码等）
- **文本查找**：在文本中查找符合特定模式的内容
- **文本替换**：替换符合特定模式的文本
- **文本提取**：从文本中提取特定信息
- **文本分割**：按照特定模式分割文本

### 1.2 Python 中的正则表达式

Python 提供了内置的 `re` 模块，用于支持正则表达式操作。这个模块提供了一系列函数，用于执行各种正则表达式操作。

要使用正则表达式，首先需要导入 `re` 模块：

```python
import re
```

## 2. 正则表达式语法

### 2.1 普通字符

普通字符包括所有可打印的 ASCII 字符（字母、数字、标点符号）和部分不可打印字符，它们在正则表达式中表示其本身。

例如：
- `a` 匹配字母 "a"
- `123` 匹配字符串 "123"
- `hello` 匹配字符串 "hello"

### 2.2 元字符

元字符是正则表达式中具有特殊含义的字符，用于定义匹配规则。

| 元字符 | 描述 |
|--------|------|
| `.` | 匹配除换行符外的任意单个字符 |
| `^` | 匹配字符串的开始 |
| `$` | 匹配字符串的结束 |
| `*` | 匹配前面的字符零次或多次 |
| `+` | 匹配前面的字符一次或多次 |
| `?` | 匹配前面的字符零次或一次 |
| `{n}` | 匹配前面的字符恰好 n 次 |
| `{n,}` | 匹配前面的字符至少 n 次 |
| `{n,m}` | 匹配前面的字符 n 到 m 次 |
| `[abc]` | 匹配字符集合中的任意一个字符 |
| `[^abc]` | 匹配除字符集合外的任意一个字符 |
| `|` | 匹配左右任意一个表达式 |
| `(...)` | 分组，将括号内的表达式作为一个整体 |
| `\` | 转义字符，用于匹配元字符本身 |

### 2.3 字符类

字符类用于匹配特定类型的字符：

| 字符类 | 描述 |
|--------|------|
| `\d` | 匹配任意数字，等价于 `[0-9]` |
| `\D` | 匹配任意非数字，等价于 `[^0-9]` |
| `\w` | 匹配任意字母、数字或下划线，等价于 `[a-zA-Z0-9_]` |
| `\W` | 匹配任意非字母、数字或下划线，等价于 `[^a-zA-Z0-9_]` |
| `\s` | 匹配任意空白字符（空格、制表符、换行符等） |
| `\S` | 匹配任意非空白字符 |
| `\b` | 匹配单词边界 |
| `\B` | 匹配非单词边界 |

### 2.4 量词

量词用于指定前面的字符或分组出现的次数：

| 量词 | 描述 | 示例 |
|------|------|------|
| `*` | 零次或多次 | `ab*c` 匹配 "ac", "abc", "abbc" 等 |
| `+` | 一次或多次 | `ab+c` 匹配 "abc", "abbc" 等，但不匹配 "ac" |
| `?` | 零次或一次 | `ab?c` 匹配 "ac" 或 "abc" |
| `{n}` | 恰好 n 次 | `ab{3}c` 匹配 "abbbc" |
| `{n,}` | 至少 n 次 | `ab{2,}c` 匹配 "abbc", "abbbc" 等 |
| `{n,m}` | n 到 m 次 | `ab{1,3}c` 匹配 "abc", "abbc", "abbbc" |

默认情况下，量词是贪婪的，即它们会尽可能多地匹配字符。在量词后添加 `?` 可以使其变为非贪婪模式，即尽可能少地匹配字符。

例如：
- `a.*b` 匹配从第一个 "a" 到最后一个 "b" 的所有内容（贪婪）
- `a.*?b` 匹配从第一个 "a" 到第一个 "b" 的内容（非贪婪）

## 3. re 模块核心函数

### 3.1 re.match()

尝试从字符串的起始位置匹配一个模式，如果匹配成功，返回一个匹配对象；否则返回 None。

```python
pattern = r'\d+'
text = '123abc456'
result = re.match(pattern, text)
print(result)  # <re.Match object; span=(0, 3), match='123'>
```

### 3.2 re.search()

在整个字符串中搜索匹配模式，返回第一个匹配的结果；如果没有找到匹配，返回 None。

```python
pattern = r'\d+'
text = 'abc123def456'
result = re.search(pattern, text)
print(result)  # <re.Match object; span=(3, 6), match='123'>
```

### 3.3 re.findall()

在整个字符串中搜索匹配模式，返回所有匹配结果的列表；如果没有找到匹配，返回空列表。

```python
pattern = r'\d+'
text = 'abc123def456ghi789'
result = re.findall(pattern, text)
print(result)  # ['123', '456', '789']
```

### 3.4 re.finditer()

在整个字符串中搜索匹配模式，返回一个迭代器，包含所有匹配对象；如果没有找到匹配，返回空迭代器。

```python
pattern = r'\d+'
text = 'abc123def456ghi789'
result = re.finditer(pattern, text)
for match in result:
    print(match.group())  # 输出: 123, 456, 789
```

### 3.5 re.split()

按照匹配的模式分割字符串，返回分割后的列表。

```python
pattern = r'\s+'
text = 'hello   world   python'
result = re.split(pattern, text)
print(result)  # ['hello', 'world', 'python']
```

### 3.6 re.sub()

替换匹配的模式为指定的字符串，返回替换后的字符串。

```python
pattern = r'\d+'
text = 'abc123def456'
result = re.sub(pattern, 'X', text)
print(result)  # 'abcXdefX'
```

### 3.7 re.subn()

与 re.sub() 类似，但返回一个元组，包含替换后的字符串和替换的次数。

```python
pattern = r'\d+'
text = 'abc123def456'
result = re.subn(pattern, 'X', text)
print(result)  # ('abcXdefX', 2)
```

### 3.8 re.compile()

编译正则表达式模式，返回一个正则表达式对象，用于提高多次使用同一模式时的效率。

```python
pattern = re.compile(r'\d+')
text = 'abc123def456'
result = pattern.findall(text)
print(result)  # ['123', '456']
```

## 4. 匹配对象的方法

当正则表达式匹配成功时，会返回一个匹配对象，这个对象提供了一些方法来获取匹配信息：

### 4.1 group()

返回匹配的字符串。如果有分组，还可以指定分组索引或名称。

```python
pattern = r'(\d+)-(\w+)'
text = '123-abc'
result = re.match(pattern, text)
print(result.group())   # '123-abc' (整个匹配)
print(result.group(1))  # '123' (第一个分组)
print(result.group(2))  # 'abc' (第二个分组)
```

### 4.2 groups()

返回所有分组匹配的字符串组成的元组。

```python
pattern = r'(\d+)-(\w+)'
text = '123-abc'
result = re.match(pattern, text)
print(result.groups())  # ('123', 'abc')
```

### 4.3 groupdict()

返回一个字典，包含所有命名分组的匹配结果。

```python
pattern = r'(?P<number>\d+)-(?P<word>\w+)'
text = '123-abc'
result = re.match(pattern, text)
print(result.groupdict())  # {'number': '123', 'word': 'abc'}
```

### 4.4 start() 和 end()

返回匹配开始和结束的位置索引。

```python
pattern = r'\d+'
text = 'abc123def'
result = re.search(pattern, text)
print(result.start())  # 3
print(result.end())    # 6
```

### 4.5 span()

返回一个元组，包含匹配开始和结束的位置索引。

```python
pattern = r'\d+'
text = 'abc123def'
result = re.search(pattern, text)
print(result.span())  # (3, 6)
```

## 5. 常用正则表达式模式

### 5.1 邮箱验证

```python
email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

# 测试
print(re.match(email_pattern, 'user@example.com'))  # 匹配成功
print(re.match(email_pattern, 'user.name+tag@example.co.uk'))  # 匹配成功
print(re.match(email_pattern, 'invalid-email'))  # 匹配失败
```

### 5.2 URL 验证

```python
url_pattern = r'^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$'

# 测试
print(re.match(url_pattern, 'https://www.example.com'))  # 匹配成功
print(re.match(url_pattern, 'http://example.com/path'))  # 匹配成功
print(re.match(url_pattern, 'www.example.com'))  # 匹配成功
print(re.match(url_pattern, 'invalid-url'))  # 匹配失败
```

### 5.3 电话号码验证

```python
# 匹配中国手机号
phone_pattern = r'^1[3-9]\d{9}$'

# 测试
print(re.match(phone_pattern, '13812345678'))  # 匹配成功
print(re.match(phone_pattern, '12345678901'))  # 匹配失败
```

### 5.4 IP 地址验证

```python
ip_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'

# 测试
print(re.match(ip_pattern, '192.168.1.1'))  # 匹配成功
print(re.match(ip_pattern, '255.255.255.0'))  # 匹配成功
print(re.match(ip_pattern, '256.0.0.1'))  # 匹配失败
```

### 5.5 日期格式验证

```python
# 匹配 YYYY-MM-DD 格式
date_pattern = r'^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$'

# 测试
print(re.match(date_pattern, '2023-12-25'))  # 匹配成功
print(re.match(date_pattern, '2023-02-29'))  # 匹配成功（但实际上2023年不是闰年）
print(re.match(date_pattern, '2023-13-01'))  # 匹配失败
```

### 5.6 密码强度验证

```python
# 至少8个字符，包含至少一个大写字母、一个小写字母、一个数字和一个特殊字符
password_pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'

# 测试
print(re.match(password_pattern, 'MyPass123!'))  # 匹配成功
print(re.match(password_pattern, 'mypass123'))  # 匹配失败（缺少大写字母和特殊字符）
print(re.match(password_pattern, 'MYPASS123'))  # 匹配失败（缺少小写字母和特殊字符）
```

## 6. 高级正则表达式技巧

### 6.1 分组与命名分组

分组用于将多个字符作为一个整体处理，使用圆括号 `()` 表示。可以使用 `group()` 方法访问分组的匹配结果。

命名分组使用 `(?P<name>pattern)` 语法，可以通过名称访问分组结果。

```python
# 普通分组
pattern = r'(\d{4})-(\d{2})-(\d{2})'
text = '2023-12-25'
result = re.match(pattern, text)
print(result.group(1))  # '2023'
print(result.group(2))  # '12'
print(result.group(3))  # '25'

# 命名分组
pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'
text = '2023-12-25'
result = re.match(pattern, text)
print(result.group('year'))  # '2023'
print(result.group('month'))  # '12'
print(result.group('day'))  # '25'
print(result.groupdict())  # {'year': '2023', 'month': '12', 'day': '25'}
```

### 6.2 非捕获分组

非捕获分组使用 `(?:pattern)` 语法，它们不会创建捕获组，主要用于分组但不保存匹配结果。

```python
pattern = r'(?:\d{4})-(\d{2})-(\d{2})'
text = '2023-12-25'
result = re.match(pattern, text)
print(result.groups())  # ('12', '25')  # 只有两个捕获组
```

### 6.3 断言

断言用于检查匹配位置的前后条件，但不包括在匹配结果中。

#### 6.3.1 正向先行断言

正向先行断言使用 `(?=pattern)` 语法，匹配后面跟着 pattern 的位置。

```python
# 匹配后面跟着 "@example.com" 的用户名
pattern = r'\w+(?=@example.com)'
text = 'user@example.com'
result = re.search(pattern, text)
print(result.group())  # 'user'
```

#### 6.3.2 负向先行断言

负向先行断言使用 `(?!pattern)` 语法，匹配后面不跟着 pattern 的位置。

```python
# 匹配后面不跟着 "@example.com" 的用户名
pattern = r'\w+(?!@example.com)'
text = 'user@gmail.com'
result = re.search(pattern, text)
print(result.group())  # 'user'
```

#### 6.3.3 正向后行断言

正向后行断言使用 `(?<=pattern)` 语法，匹配前面是 pattern 的位置。

```python
# 匹配前面是 "https://" 的域名
pattern = r'(?<=https://)\w+\.\w+'
text = 'https://www.example.com'
result = re.search(pattern, text)
print(result.group())  # 'www.example'
```

#### 6.3.4 负向后行断言

负向后行断言使用 `(?<!pattern)` 语法，匹配前面不是 pattern 的位置。

```python
# 匹配前面不是 "https://" 的域名
pattern = r'(?<!https://)\w+\.\w+'
text = 'http://www.example.com'
result = re.search(pattern, text)
print(result.group())  # 'www.example'
```

### 6.4 贪婪与非贪婪匹配

默认情况下，量词是贪婪的，它们会尽可能多地匹配字符。在量词后添加 `?` 可以使其变为非贪婪模式。

```python
text = '<div>内容1</div><div>内容2</div>'

# 贪婪匹配（匹配整个字符串）
greedy_pattern = r'<div>.*</div>'
result = re.search(greedy_pattern, text)
print(result.group())  # '<div>内容1</div><div>内容2</div>'

# 非贪婪匹配（只匹配第一个 div）
non_greedy_pattern = r'<div>.*?</div>'
result = re.search(non_greedy_pattern, text)
print(result.group())  # '<div>内容1</div>'
```

### 6.5 多行模式

多行模式使用 `re.MULTILINE` 或 `re.M` 标志，使 `^` 和 `$` 匹配每行的开始和结束。

```python
text = 'Line 1\nLine 2\nLine 3'

# 单行模式（默认）
pattern = r'^Line'
result = re.findall(pattern, text)
print(result)  # ['Line']  # 只匹配第一行的开始

# 多行模式
pattern = r'^Line'
result = re.findall(pattern, text, re.MULTILINE)
print(result)  # ['Line', 'Line', 'Line']  # 匹配每行的开始
```

### 6.6 忽略大小写

忽略大小写使用 `re.IGNORECASE` 或 `re.I` 标志，使匹配不区分大小写。

```python
text = 'Hello World'

# 默认模式
pattern = r'hello'
result = re.search(pattern, text)
print(result)  # None  # 匹配失败

# 忽略大小写模式
pattern = r'hello'
result = re.search(pattern, text, re.IGNORECASE)
print(result.group())  # 'Hello'  # 匹配成功
```

## 7. 实际应用案例

### 7.1 提取网页中的链接

```python
import re
import requests

# 获取网页内容
url = 'https://www.example.com'
response = requests.get(url)
content = response.text

# 提取所有链接
link_pattern = r'href=[\'"](.*?)[\'"]'
links = re.findall(link_pattern, content)

# 打印结果
for link in links:
    print(link)
```

### 7.2 替换文本中的敏感信息

```python
import re

text = '用户信息：姓名：张三，手机号：13812345678，邮箱：zhangsan@example.com'

# 替换手机号中间4位为*
phone_pattern = r'(1[3-9])(\d{4})(\d{4})'
result = re.sub(phone_pattern, r'\1****\3', text)
print(result)  # '用户信息：姓名：张三，手机号：138****5678，邮箱：zhangsan@example.com'

# 替换邮箱@前3位为*
email_pattern = r'(\w{3})(\w*)(@.*)' 
result = re.sub(email_pattern, r'***\2\3', result)
print(result)  # '用户信息：姓名：张三，手机号：138****5678，邮箱：***gsan@example.com'
```

### 7.3 解析日志文件

```python
import re

# 日志文件内容示例
log_content = '''
2023-12-25 10:30:45 INFO User login successful: user123
2023-12-25 10:31:20 ERROR Database connection failed: Connection refused
2023-12-25 10:32:15 WARNING Disk space low: 10% remaining
''' 

# 解析日志条目
log_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) (.+)'
logs = re.findall(log_pattern, log_content)

# 打印解析结果
for log in logs:
    timestamp, level, message = log
    print(f"时间: {timestamp}, 级别: {level}, 消息: {message}")
```

### 7.4 验证和格式化文本

```python
import re

# 验证并格式化身份证号
id_card_pattern = r'^(\d{6})(\d{4})(\d{2})(\d{2})(\d{3})([0-9Xx])$'

id_card = '110101199001011234'
result = re.match(id_card_pattern, id_card)

if result:
    # 格式化身份证号
    formatted_id = f"{result.group(1)} {result.group(2)} {result.group(3)} {result.group(4)} {result.group(5)}{result.group(6)}"
    print(f"格式化后的身份证号: {formatted_id}")
    
    # 提取出生日期
    birth_date = f"{result.group(2)}-{result.group(3)}-{result.group(4)}"
    print(f"出生日期: {birth_date}")
else:
    print("无效的身份证号")
```

## 8. 性能优化和最佳实践

### 8.1 编译正则表达式

对于频繁使用的正则表达式，应该使用 `re.compile()` 编译，以提高性能。

```python
# 不推荐
for text in texts:
    re.search(r'\d+', text)

# 推荐
pattern = re.compile(r'\d+')
for text in texts:
    pattern.search(text)
```

### 8.2 避免回溯

回溯是指正则表达式引擎在匹配失败时尝试其他可能的匹配路径，这会导致性能下降。应避免使用容易导致大量回溯的模式。

例如，避免使用 `(a+)+b` 这样的模式，它会导致指数级的回溯。

### 8.3 使用非捕获分组

对于不需要保存匹配结果的分组，使用非捕获分组 `(?:pattern)` 可以提高性能。

### 8.4 限制匹配范围

使用 `^` 和 `$` 限制匹配范围，可以减少不必要的匹配尝试。

### 8.5 优先使用简单模式

对于简单的字符串操作，优先使用字符串方法（如 `str.find()`, `str.replace()`, `str.split()` 等），它们比正则表达式更快。

### 8.6 使用原始字符串

在定义正则表达式模式时，始终使用原始字符串（以 `r` 开头），以避免转义字符的问题。

```python
# 不推荐
pattern = '\\d+'

# 推荐
pattern = r'\d+'
```

## 9. 常见问题和解决方案

### 9.1 转义字符问题

问题：正则表达式中的特殊字符需要转义，但转义字符本身也需要转义，导致模式难以阅读。

解决方案：使用原始字符串（以 `r` 开头），这样不需要转义反斜杠。

```python
# 不推荐
pattern = '\\d+\\.\\d+'

# 推荐
pattern = r'\d+\.\d+'
```

### 9.2 匹配失败问题

问题：正则表达式不匹配预期的文本。

解决方案：
1. 检查正则表达式语法是否正确
2. 使用 `re.DEBUG` 标志查看正则表达式的编译过程
3. 使用在线正则表达式测试工具（如 regex101.com）测试模式

```python
pattern = re.compile(r'\d+', re.DEBUG)
```

### 9.3 性能问题

问题：正则表达式匹配速度慢。

解决方案：
1. 编译正则表达式
2. 避免回溯
3. 使用非捕获分组
4. 限制匹配范围
5. 考虑使用更简单的方法

### 9.4 贪婪匹配问题

问题：正则表达式匹配了比预期更多的文本。

解决方案：使用非贪婪量词（在量词后添加 `?`）。

### 9.5 多行文本匹配问题

问题：正则表达式不能正确匹配多行文本。

解决方案：使用 `re.MULTILINE` 标志匹配每行的开始和结束，或使用 `re.DOTALL` 标志使 `.` 匹配换行符。

## 10. 总结

正则表达式是一种强大的文本处理工具，Python 的 `re` 模块提供了全面的支持。通过掌握正则表达式的语法和 `re` 模块的功能，可以有效地处理各种文本操作任务。

学习正则表达式需要时间和实践，建议从简单的模式开始，逐步学习更复杂的概念。通过不断练习和应用，你将能够熟练地使用正则表达式解决各种文本处理问题。

## 11. 学习资源

- [Python 官方文档 - re 模块](https://docs.python.org/3/library/re.html)
- [正则表达式教程 - RegexOne](https://regexone.com/)
- [正则表达式测试工具 - regex101](https://regex101.com/)
- [正则表达式速查表](https://www.regexbuddy.com/regex.html)