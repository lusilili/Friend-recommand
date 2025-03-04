import pandas as pd

# 读取好友推荐表
df = pd.read_excel('recommendations.xlsx')

# 输入用户ID
user_id = input('请输入用户ID：')

# 查询用户推荐的好友列表
recommendations = df.loc[user_id]

# 打印推荐结果
print('推荐结果：')
print(recommendations)