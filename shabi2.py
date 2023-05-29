import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 读取并打印数据的前5行
print(pd.read_csv('anonymous-msweb.csv', nrows=10))

# 1. 数据预处理
# 读取数据集并添加列名
data = pd.read_csv('anonymous-msweb.csv', header=None,names=[ 'column_1','user_id','column_3','column_4','page_id','column_6'],nrows=1000)
# 将 'page_id' 列中的 0 值替换为 'none'
data = data.fillna('none')
data['page_id'] = data['page_id'].replace(0, 'none')
# 提取用户浏览记录
user_records = data.groupby('user_id')['page_id'].apply(list)
print("user_records:",user_records)
print("user_id:",data['user_id'])
print("page_id:",data['page_id'])
# 2. 数据探索性分析
# 分析最常被访问的页面
most_visited_pages = data['page_id'].value_counts().head(10)

# 分析页面访问量分布
page_visit_distribution = data['page_id'].value_counts()

# 3. 关联规则挖掘

# 将用户浏览记录进行去重并进行独热编码

encoded_records = user_records.apply(lambda x: pd.Series(1, index=set(x)))
encoded_records=encoded_records.fillna(0)
# 将 'none' 值替换为 0
encoded_records = encoded_records.replace('none', 0)


print("encoded_records:",encoded_records)
# 使用Apriori算法计算频繁项集
frequent_itemsets = apriori(encoded_records, min_support=0.001, use_colnames=True)
print('frequent_itemsets:',frequent_itemsets)
# 根据频繁项集计算关联规则
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)

# 4. 结果评估
# 计算关联规则的支持度、置信度和提升度
rules['support'] = rules['support'] * len(user_records)
rules['confidence'] = rules['confidence']
rules['lift'] = rules['lift']

# 5. 结果分析与应用
# 分析得到的关联规则，为网站提供导航结构优化建议
navigation_suggestions = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

# 输出结果
print("最常被访问的页面：")
print(most_visited_pages)
print("\n页面访问量分布：")
print(page_visit_distribution)
print("\n导航结构优化建议：")
print(navigation_suggestions)
