以下是分步解决方案，修复`KeyError: 'sentiment'`错误：

---

### **1. 检查数据预处理流程**
```python
# 确认预处理步骤包含情感分析
print("预处理后的列名：", df.columns.tolist())

# 如果缺少'sentiment'列，重新生成
if 'sentiment' not in df.columns:
    # 重新执行情感分析步骤
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    sid = SentimentIntensityAnalyzer()
    df['sentiment'] = df['text'].apply(
        lambda x: 'positive' if sid.polarity_scores(x)['compound'] > 0.2 
        else 'negative' if sid.polarity_scores(x)['compound'] < -0.2 
        else 'neutral'
    )
    # 重新保存数据
    df.to_csv('preprocessed_news.csv', index=False)
```

---

### **2. 验证数据加载**
```python
# 重新加载数据并检查
df = pd.read_csv('preprocessed_news.csv')
print("加载后的列名：", df.columns.tolist())
print("前5行情感标签：\n", df['sentiment'].head())
```

---

### **3. 完整修复代码**
```python
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 加载数据
df = pd.read_csv('preprocessed_news.csv')

# 检查并修复缺失列
if 'sentiment' not in df.columns:
    # 生成情感标签
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()
    
    def get_news_sentiment(text):
        scores = sid.polarity_scores(str(text))  # 强制转换为字符串
        return 'positive' if scores['compound'] > 0.2 else 'negative' if scores['compound'] < -0.2 else 'neutral'
    
    df['sentiment'] = df['text'].apply(get_news_sentiment)
    df.to_csv('preprocessed_news.csv', index=False)
    print("情感标签已重新生成并保存")
else:
    print("情感标签列已存在")

# 验证结果
print("情感标签分布：\n", df['sentiment'].value_counts())
```

---

### **4. 关键修复点**
1. **列存在性检查**：
   ```python
   if 'sentiment' not in df.columns:
       # 重新生成逻辑
   ```

2. **防御性编程**：
   ```python
   str(text)  # 确保输入为字符串
   ```

3. **数据持久化**：
   ```python
   df.to_csv('preprocessed_news.csv', index=False)
   ```

---

### **5. 运行结果示例**
```
预处理后的列名： ['text', 'category']
情感标签已重新生成并保存
情感标签分布：
 neutral     213456
positive     156789
negative      65432
Name: sentiment, dtype: int64
```

---

### **6. 后续步骤**
1. 确保所有成员使用最新版本的`preprocessed_news.csv`
2. 在特征工程前添加列验证：
   ```python
   assert 'sentiment' in df.columns, "数据必须包含情感标签列"
   ```

通过此修复，可以彻底解决因缺失`sentiment`列导致的`KeyError`，确保后续分析顺利进行。
