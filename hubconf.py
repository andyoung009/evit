# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from models import *

dependencies = ["torch", "torchvision", "timm"]

# 这段代码是一个 Python 模块或脚本的头部，包括两部分内容：
# 1. `from models import *` 表示从名为 `models` 的模块中导入所有内容（函数、类、变量等），并将其添加到当前命名空间中。这意味着在后面的代码中可以直接使用 `models` 模块中定义的所有内容，而无需使用 `models.` 前缀。
# 2. `dependencies = ["torch", "torchvision", "timm"]` 定义了一个名为 `dependencies` 的列表，其中包含三个字符串元素 `"torch"`、`"torchvision"` 和 `"timm"`。这个列表可能是用于记录当前模块或脚本所依赖的第三方库或模块。
# 通常情况下，Python 模块或脚本的头部会包含导入必要的库和模块的语句，以及一些元数据和注释信息。这些信息可以帮助其他开发者更容易地理解和使用该模块或脚本，并在必要时对其进行修改和维护。