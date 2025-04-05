# Recommendation System

```js
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

class NewsRecommender:
    def __init__(self, df):
        self.df = df
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化BERT tokenizer和模型
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.num_labels = len(df['category'].unique())
        
        # 初始化情感分析模型
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",  # 使用金融新闻专用的BERT模型
            device=0 if torch.cuda.is_available() else -1
        )
        
        # 初始化BERT分类模型
        self.bert_model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=self.num_labels
        ).to(self.device)
        
        # 标签编码器
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(df['category'])
        
    def prepare_data(self, texts, max_length=200, batch_size=32):
        # Tokenize数据
        encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # 创建数据集
        dataset = TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask']
        )
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        return dataloader
    
    def get_bert_embeddings(self, texts):
        """获取BERT的文本嵌入"""
        dataloader = self.prepare_data(texts)
        embeddings = []
        
        self.bert_model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                
                outputs = self.bert_model(input_ids, attention_mask=attention_mask)
                # 使用最后一层的[CLS]token的输出作为文本表示
                embeddings.append(outputs.hidden_states[-1][:, 0, :].cpu().numpy())
        
        return np.vstack(embeddings)
    
    def get_sentiment_scores(self, texts):
        """获取情感分析分数"""
        sentiments = []
        for text in texts:
            result = self.sentiment_analyzer(text[:512])[0]  # 限制文本长度
            sentiments.append(result['score'])
        return np.array(sentiments)
    
    def train_bert(self, train_texts, train_labels, epochs=3):
        """训练BERT分类模型"""
        train_dataloader = self.prepare_data(train_texts)
        optimizer = torch.optim.AdamW(self.bert_model.parameters(), lr=2e-5)
        
        self.bert_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_dataloader:
                optimizer.zero_grad()
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = torch.tensor(train_labels).to(self.device)
                
                outputs = self.bert_model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader):.4f}")
    
    def recommend(self, article_id, n_recommendations=5):
        """基于BERT嵌入和情感分析进行推荐"""
        # 获取所有文章的BERT嵌入
        all_embeddings = self.get_bert_embeddings(self.df['cleaned_text'])
        
        # 获取所有文章的情感分数
        all_sentiments = self.get_sentiment_scores(self.df['cleaned_text'])
        
        # 计算相似度
        target_embedding = all_embeddings[article_id]
        target_sentiment = all_sentiments[article_id]
        
        # 计算BERT嵌入的余弦相似度
        bert_similarities = cosine_similarity([target_embedding], all_embeddings)[0]
        
        # 计算情感相似度
        sentiment_similarities = 1 - np.abs(all_sentiments - target_sentiment)
        
        # 综合相似度分数
        combined_scores = 0.7 * bert_similarities + 0.3 * sentiment_similarities
        
        # 获取推荐文章的索引
        recommended_indices = np.argsort(combined_scores)[-n_recommendations-1:-1][::-1]
        
        # 获取推荐结果
        recommendations = []
        for idx in recommended_indices:
            recommendations.append({
                'article_id': idx,
                'title': self.df.iloc[idx]['headline'],
                'category': self.df.iloc[idx]['category'],
                'sentiment_score': all_sentiments[idx],
                'similarity_score': combined_scores[idx]
            })
        
        return recommendations

# 使用示例
print("Initializing recommendation system...")
recommender = NewsRecommender(df)

# 训练BERT模型
print("\nTraining BERT model...")
train_texts = df['cleaned_text']
train_labels = recommender.label_encoder.transform(df['category'])
recommender.train_bert(train_texts, train_labels)

# 为第一篇文章生成推荐
article_id = 0
recommendations = recommender.recommend(article_id)

print("\nOriginal article:")
print(f"Title: {df.iloc[article_id]['headline']}")
print(f"Category: {df.iloc[article_id]['category']}")
print(f"Sentiment: {recommender.get_sentiment_scores([df.iloc[article_id]['cleaned_text']])[0]:.2f}")

print("\nRecommended articles:")
for rec in recommendations:
    print(f"\nTitle: {rec['title']}")
    print(f"Category: {rec['category']}")
    print(f"Sentiment: {rec['sentiment_score']:.2f}")
    print(f"Similarity: {rec['similarity_score']:.2f}")
```


```js
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def visualize_recommendations(article_id, recommendations, df):
    # 创建图形
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 情感分布对比
    plt.subplot(2, 2, 1)
    original_sentiment = recommender.get_sentiment_scores([df.iloc[article_id]['cleaned_text']])[0]
    rec_sentiments = [rec['sentiment_score'] for rec in recommendations]
    
    plt.bar(['Original'] + [f'Rec {i+1}' for i in range(len(recommendations))],
            [original_sentiment] + rec_sentiments)
    plt.title('Sentiment Comparison')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45)
    
    # 2. 相似度分数分布
    plt.subplot(2, 2, 2)
    rec_similarities = [rec['similarity_score'] for rec in recommendations]
    plt.bar(range(1, len(recommendations) + 1), rec_similarities)
    plt.title('Similarity Scores')
    plt.xlabel('Recommendation Rank')
    plt.ylabel('Similarity Score')
    
    # 3. 类别分布
    plt.subplot(2, 2, 3)
    original_category = df.iloc[article_id]['category']
    rec_categories = [rec['category'] for rec in recommendations]
    
    category_counts = pd.Series([original_category] + rec_categories).value_counts()
    category_counts.plot(kind='bar')
    plt.title('Category Distribution')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # 4. 关键词词云
    plt.subplot(2, 2, 4)
    text = ' '.join([df.iloc[article_id]['cleaned_text']] + 
                    [df.iloc[rec['article_id']]['cleaned_text'] for rec in recommendations])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Keywords in Articles')
    
    plt.tight_layout()
    plt.show()

# 可视化推荐结果
visualize_recommendations(0, recommendations, df)

# 情感分布可视化
plt.figure(figsize=(10, 6))
all_sentiments = recommender.get_sentiment_scores(df['cleaned_text'])
sns.histplot(all_sentiments, bins=50)
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Count')
plt.show()

# 类别-情感关系可视化
plt.figure(figsize=(12, 6))
df['sentiment_score'] = all_sentiments
sns.boxplot(data=df, x='category', y='sentiment_score')
plt.title('Sentiment Distribution by Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Category classification
TF-IDF + SVM
BERT


```js
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

print("Training TF-IDF + SVM model...")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], 
    df['category'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['category']
)

# 创建pipeline
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', LinearSVC(random_state=42))
])

# 训练模型
svm_pipeline.fit(X_train, y_train)

# 预测和评估
y_pred_svm = svm_pipeline.predict(X_test)

# 打印分类报告
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))

# 保存结果用于可视化
svm_results = {
    'y_true': y_test,
    'y_pred': y_pred_svm,
    'model_name': 'SVM'
}
```

结果

```js
Training TF-IDF + SVM model...

Classification Report:
                precision    recall  f1-score   support

          ARTS       0.29      0.18      0.22       302
ARTS & CULTURE       0.27      0.16      0.20       268
  BLACK VOICES       0.44      0.32      0.37       917
      BUSINESS       0.47      0.41      0.44      1198
       COLLEGE       0.41      0.37      0.39       229
        COMEDY       0.56      0.39      0.46      1080
         CRIME       0.49      0.54      0.52       712
CULTURE & ARTS       0.41      0.23      0.29       215
       DIVORCE       0.75      0.68      0.71       685
     EDUCATION       0.41      0.34      0.37       203
 ENTERTAINMENT       0.57      0.70      0.63      3473
   ENVIRONMENT       0.42      0.27      0.33       289
         FIFTY       0.35      0.16      0.22       280
  FOOD & DRINK       0.57      0.69      0.62      1268
     GOOD NEWS       0.26      0.17      0.21       280
         GREEN       0.37      0.31      0.34       524
HEALTHY LIVING       0.36      0.17      0.23      1339
 HOME & LIVING       0.62      0.67      0.65       864
        IMPACT       0.37      0.25      0.29       697
 LATINO VOICES       0.57      0.27      0.37       226
         MEDIA       0.50      0.36      0.42       589
         MONEY       0.44      0.37      0.40       351
     PARENTING       0.50      0.60      0.55      1758
       PARENTS       0.40      0.21      0.27       791
      POLITICS       0.70      0.83      0.76      7121
  QUEER VOICES       0.73      0.65      0.69      1269
      RELIGION       0.54      0.49      0.52       515
       SCIENCE       0.47      0.43      0.45       441
        SPORTS       0.58      0.64      0.61      1015
         STYLE       0.40      0.20      0.27       451
STYLE & BEAUTY       0.70      0.79      0.74      1963
         TASTE       0.30      0.12      0.17       419
          TECH       0.47      0.43      0.45       421
 THE WORLDPOST       0.46      0.38      0.41       733
        TRAVEL       0.64      0.76      0.69      1980
     U.S. NEWS       0.37      0.11      0.17       275
      WEDDINGS       0.76      0.76      0.76       731
    WEIRD NEWS       0.33      0.23      0.27       555
      WELLNESS       0.55      0.75      0.64      3589
         WOMEN       0.38      0.26      0.31       714
    WORLD NEWS       0.39      0.31      0.35       660
     WORLDPOST       0.36      0.26      0.30       516

      accuracy                           0.57     41906
     macro avg       0.47      0.41      0.43     41906
  weighted avg       0.55      0.57      0.55     41906
```

```js
import seaborn as sns
import matplotlib.pyplot as plt

# 混淆矩阵可视化
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=svm_pipeline.classes_,
            yticklabels=svm_pipeline.classes_)
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 每个类别的准确率可视化
class_report = classification_report(y_test, y_pred_svm, output_dict=True)
class_df = pd.DataFrame(class_report).transpose()
plt.figure(figsize=(10, 6))
class_df['precision'].iloc[:-3].plot(kind='bar')
plt.title('Precision by Category - SVM')
plt.xlabel('Category')
plt.ylabel('Precision')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/fa94818b-84aa-4f0a-be64-55feb2b797f7)


![image](https://github.com/user-attachments/assets/fce4f91b-cc02-46cd-b8a3-cbadc58e9a08)


```js
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.preprocessing import LabelEncoder

print("Training BERT model...")

# 初始化tokenizer和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
num_labels = len(df['category'].unique())
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
                                                     num_labels=num_labels)

# 标签编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['category'])

# 准备数据
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], 
    y_encoded,
    test_size=0.2, 
    random_state=42,
    stratify=y_encoded
)

# Tokenize数据
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=512)

# 创建数据加载器
train_dataset = TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(y_train)
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 训练循环
model.train()
for epoch in range(3):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# 评估
model.eval()
test_dataset = TensorDataset(
    torch.tensor(test_encodings['input_ids']),
    torch.tensor(test_encodings['attention_mask'])
)
test_loader = DataLoader(test_dataset, batch_size=16)

y_pred_bert = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)
        y_pred_bert.extend(predictions.cpu().numpy())

# 转换回原始标签
y_pred_bert = label_encoder.inverse_transform(y_pred_bert)
y_test_original = label_encoder.inverse_transform(y_test)

print("\nClassification Report:")
print(classification_report(y_test_original, y_pred_bert))

# 保存结果用于可视化
bert_results = {
    'y_true': y_test_original,
    'y_pred': y_pred_bert,
    'model_name': 'BERT'
}
```


```js
Training BERT model...
tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]
vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]
tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]
config.json:   0%|          | 0.00/570 [00:00<?, ?B/s]
model.safetensors:   0%|          | 0.00/440M [00:00<?, ?B/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

![image](https://github.com/user-attachments/assets/5c0520fa-fe7d-4e50-9270-1ee7dc156f48)


