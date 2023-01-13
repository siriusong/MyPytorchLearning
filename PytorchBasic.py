import numpy as np
import torch
# 获取张量的数据类型
torch.tensor([1.2, 3.4]).dtype
# 更改默认类型
torch.set_default_tensor_type(torch.DoubleTensor)
torch.tensor([1.2, 3.4]).dtype
torch.set_default_tensor_type(torch.FloatTensor)
# 获取默认的数据类型
torch.get_default_dtype()

# list可以通过torch.tensor来构造张量
lst = [[1.0, 1.0], [2, 2]]
type(lst)
A = torch.tensor(lst)
A
# 张量的属性
A.shape
A.size()
A.numel()

# 计算梯度的张量
B = torch.tensor((1, 2, 3), dtype=torch.float32, requires_grad=True)
B

# 计算sum(B**2)的梯度
y = B.pow(2).sum()
y.backward()
B.grad

# 创建具有特定大小的张量
D = torch.Tensor(2, 3)
D
# 根据已有数据创建张量
C = torch.Tensor(lst)
C
# 创建一个与D相同大小和类型的全1张量
E = torch.ones_like(D)
E

# 张量和numpy数组相互转换
F = np.ones((3, 3))
F
Ftensor = torch.as_tensor(F)
Ftensor
FFtendor = Ftensor.float()
FFtendor.dtype
Ftensor.numpy()

# 随机数生成张量
torch.manual_seed(123)
A = torch.normal(mean=0.0, std=torch.tensor(1.0))
A

# 其他生成张量的函数
torch.arange(start=0, end=10, step=2)
torch.linspace(start=1, end=10, steps=5)


# 设置张量形状的大小
A = torch.arange(12.0).reshape(4, 3)
A

# 输入一个张量，然后，改变形状,给出一个行，然后用-1来代替列
torch.reshape(input=A, shape=(2, -1))
torch.reshape(input=A, shape=(3, -1))

A.resize_(2, 6)

# 插入新张量
A = torch.arange(12.0).reshape(2, 6)
B = torch.unsqueeze(A, dim=0)  # 在第一个维度前插入
B.shape

C = B.unsqueeze(dim=3)
C.shape
# 移除指定维度为1的维度
E = torch.squeeze(C, dim=0)
E.shape
# 用expand方法拓展张量
A = torch.arange(3)
A
B = A.expand(3, -1)
B

# 按照某个张量的形状进行拓展
C = torch.arange(6).reshape(2, 3)
B = A.expand_as(C)
B
# 把一个张量看作整体进行重复填充
D = B.repeat(1, 2, 2)
D
D.shape

# 获取张量的元素
A = torch.arange(12).reshape(1, 3, 4)
A[0]
# 第0维度，前两行，列全都要
A[0, 0:2, :]
# 第0维度，最后一行，-1到-4列
A[0, -1, -4:-1]
# 根据bool值进行索引
B = -A
# 当A>5为true时返回A对应位置值，为false返回B的值
torch.where(A > 5, A, B)
# 获取大于5的元素
A[A > 5]
# 获取张量的上三角或下三角的部分
torch.tril(A, diagonal=0)
torch.tril(A, diagonal=1)
torch.triu(A, diagonal=0)
# 获得张量对角线的元素
C = A.reshape(3, 4)
C
torch.diag(C, diagonal=0)
torch.diag(C, diagonal=1)
torch.diag(torch.tensor([1, 2, 3]))
# 张量的拼接和拆分
A = torch.arange(6.0).reshape(2, 3)
B = torch.linspace(0, 10, 6).reshape(2, 3)
C = torch.cat((A, B), dim=0)
C
D = torch.cat((A, B), dim=1)
D
# 也可以拼接三个张量
E = torch.cat((A[:, 1:2], A, B), dim=1)
E
# torch.stack()一样的效果
