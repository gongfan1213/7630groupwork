# Group O COMP7630 Project
# 1.Dataset

This dataset contains around 240k news headlines from 2012 to 2025 from HuffPost. Based on kaggle dataset (2012-2022) and new post scrapped directly from Huffpost.com (2022 - 2025). It's one of the biggest news datasets and can serve as a benchmark for a variety of tasks.

Each record in the dataset consists of the following attributes:

category: category in which the article was published.

headline: the headline of the news article.

authors: list of authors who contributed to the article.

link: link to the original news article.

short_description: Abstract of the news article.

date: publication date of the article.

example:

```js
{
  "link": "https://www.huffpost.com/entry/covid-boosters-uptake-us_n_632d719ee4b087fae6feaac9",
  "headline": "Over 4 Million Americans Roll Up Sleeves For Omicron-Targeted COVID Boosters",
  "category": "U.S. NEWS",
  "short_description": "Health experts said it is too early to predict whether demand would match up with the 171 million doses of the new boosters the U.S. ordered for the fall.",
  "authors": "Carla K. Johnson, AP",
  "date": "2022-09-23"
}
```
 # Environment Setup if running on Google Colab


```js
from google.colab import drive
import pandas as pd


drive.mount('/content/drive')
json_path = '/content/drive/MyDrive/data/News_Category_Dataset_v3.json'
df = pd.read_json(json_path, lines=True)
```

# Environment Setup local

```js
import pandas as pd


json_path = "./News_Category_Dataset_v3.json"

df = pd.read_json(json_path, lines=True)
```

```js
# Verify the data
df.head()
```

# 2.Dataset Analysis and Data Preprocessing.

Data Preprocessing

Sentence segmentation and tokenization
Remove URLs and email addresses
Stop word removal
Filter out short sentences
Data Analysis (Analyze before and after preprocessing)

Analyze text length distribution
Analyze class distribution
Calculate keywords for each category
Generate word clouds for visualization

```js
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import nltk

# Init
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

extra_stopwords = ['said', 'say', 'would', 'could', 'also', 'one', 'two', 'make', 'may',
                   'u', 'time', 'new']
stop_words.update(extra_stopwords)

print(stop_words)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

```

```js
# 1. Sentence segmentation and tokenization

df.dropna(subset=['headline', 'short_description'], inplace=True)

df['text'] = df['headline'] + ' ' + df['short_description'].fillna('')

df['cleaned_text'] = df['text'].apply(preprocess_text)
```


```js
# 2. display the distribution of text length and category

import matplotlib.pyplot as plt
import seaborn as sns

df['text_length'] = df['text'].str.len()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 24))

# Plot 1: Distribution of text length
sns.histplot(data=df, x='text_length', bins=30, ax=ax1)
ax1.set_title('Distribution of Text Length')
ax1.set_xlabel('Text Length (characters)')
ax1.set_ylabel('Count')

# Plot 2: Box plot of text length by category
sns.boxplot(data=df, x='category', y='text_length', ax=ax2)
ax2.set_title('Text Length Distribution by Category')
ax2.set_xlabel('Category')
ax2.set_ylabel('Text Length (characters)')
plt.xticks(rotation=45, ha='right')  # Fixed: using plt.xticks instead of ax2.xticks


# Plot 3:
category_counts = df['category'].value_counts()
sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax3)
ax3.set_title('Distribution of News Categories')
ax3.set_xlabel('Category')
ax3.set_ylabel('Count')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 4: word cloud
from wordcloud import WordCloud
text = ' '.join(df['cleaned_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
ax4.imshow(wordcloud)
ax4.set_title('Word Cloud of News Content')
ax4.axis('off')

plt.tight_layout()
plt.show()
```

```js
# filter out low frequency words and short words

all_words = ' '.join(df['cleaned_text']).split()
word_counts = Counter(all_words)
min_freq = 5
min_length = 3
df['filtered_text'] = df['cleaned_text'].apply(
    lambda x: ' '.join([word for word in x.split()
    if word_counts[word] >= min_freq and len(word) >= min_length])
)

top_categories = df['category'].value_counts().head(15).index
df = df[df['category'].isin(top_categories)]

# 处理类别不平衡（示例：过采样）
from imblearn.over_sampling import RandomOverSampler
X = df['filtered_text']
y = df['category']
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X.to_frame(), y)
df_balanced = pd.DataFrame({'text': X_resampled['filtered_text'], 'category': y_resampled})

df_balanced.to_csv('preprocessed_news.csv', index=False)
```

```js
# display distribution after preprocessing

df_balanced['text_length'] = df_balanced['text'].str.len()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

# Plot 1: Distribution of category
sns.histplot(data=df_balanced, x='category', bins=30, ax=ax1)
ax1.set_title('Distribution of Category')
ax1.set_xlabel('Category')
ax1.set_ylabel('Count')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Top 20 words bar chart
all_words = ' '.join(df_balanced['text']).split()
word_counts = Counter(all_words)
top_words = word_counts.most_common(20)
words, counts = zip(*top_words)

sns.barplot(x=list(counts), y=list(words), palette='viridis', ax=ax2)
ax2.set_title('Top 20 Most Frequent Words')
ax2.set_xlabel('Count')
ax2.set_ylabel('Words')


# Plot 3: Overall word cloud visualization
text = ' '.join(df_balanced['text'])
wordcloud = WordCloud(width=800, height=400,
                     background_color='white',
                     max_words=100,
                     colormap='viridis').generate(text)

ax3.imshow(wordcloud)
ax3.set_title('Overall Word Cloud')
ax3.axis('off')

plt.tight_layout()
plt.show()
```

```js
# Add dropdown to show top 15 frequent words by category
from ipywidgets import interact, Dropdown

def show_category_top_words(category):
    plt.figure(figsize=(10, 6))

    # Get text for selected category
    category_text = ' '.join(df_balanced[df_balanced['category'] == category]['text'])

    # Count word frequencies
    words = category_text.split()
    word_counts = Counter(words)

    # Get top 15 words
    top_words = word_counts.most_common(15)
    words, counts = zip(*top_words)

    # Create bar chart
    sns.barplot(x=list(counts), y=list(words), palette='viridis')
    plt.title(f'Top 15 Most Frequent Words in Category: {category}')
    plt.xlabel('Count')
    plt.ylabel('Words')
    plt.tight_layout()
    plt.show()

# This will create an interactive dropdown in the notebook
interact(show_category_top_words,
         category=Dropdown(options=sorted(df_balanced['category'].unique()),
                          description='Category:'))
```


# Topic Modeling

Topic Modeling
LDA

```js
# init

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from bertopic import BERTopic
import pyLDAvis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Topic number
n_topics = 10
```


```js
# 2. LDA (Latent Dirichlet Allocation)
print("Training LDA model...")

# 创建词频矩阵
count_vectorizer = CountVectorizer(max_features=5000)
count_matrix = count_vectorizer.fit_transform(df['cleaned_text'])

# 训练LDA模型
lda_model = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    max_iter=50
)
lda_output = lda_model.fit_transform(count_matrix)

# 获取主题词
feature_names = count_vectorizer.get_feature_names_out()
lda_topics = {}
for topic_idx, topic in enumerate(lda_model.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
    lda_topics[f"Topic {topic_idx+1}"] = top_words
    print(f"Topic {topic_idx+1}: {', '.join(top_words)}")
```

```js
def visualize_lda_results(lda_model, count_vectorizer, lda_output, n_words=10):
    # 1. 主题-词语分布可视化
    feature_names = count_vectorizer.get_feature_names_out()

    # 创建一个大图，包含所有主题
    n_topics = len(lda_model.components_)
    n_rows = (n_topics + 4) // 5  # 每行5个主题
    fig = plt.figure(figsize=(20, 4*n_rows))

    for i, topic in enumerate(lda_model.components_):
        plt.subplot(n_rows, 5, i + 1)
        top_words_idx = topic.argsort()[:-n_words-1:-1]
        top_words = [feature_names[idx] for idx in top_words_idx]
        top_weights = topic[top_words_idx]

        plt.barh(top_words, top_weights)
        plt.title(f'Topic {i+1}')
        plt.xlabel('Weight')

    plt.tight_layout()
    plt.show()

    # 2. 主题分布热力图
    plt.figure(figsize=(12, 8))
    topic_term_matrix = pd.DataFrame(
        lda_model.components_,
        columns=feature_names,
        index=[f'Topic {i+1}' for i in range(n_topics)]
    )
    sns.heatmap(topic_term_matrix.iloc[:, :30], cmap='YlOrRd')
    plt.title('Topic-Term Heatmap')
    plt.xlabel('Terms')
    plt.ylabel('Topics')
    plt.show()

    # 3. 文档-主题分布
    doc_topics = pd.DataFrame(
        lda_output,
        columns=[f'Topic {i+1}' for i in range(n_topics)]
    )

    plt.figure(figsize=(10, 6))
    doc_topics.mean().plot(kind='bar')
    plt.title('Average Topic Distribution Across Documents')
    plt.xlabel('Topics')
    plt.ylabel('Average Weight')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 4. 主题相关性
    plt.figure(figsize=(10, 8))
    topic_corr = np.corrcoef(lda_output.T)
    sns.heatmap(
        topic_corr,
        annot=True,
        cmap='coolwarm',
        xticklabels=[f'T{i+1}' for i in range(n_topics)],
        yticklabels=[f'T{i+1}' for i in range(n_topics)]
    )
    plt.title('Topic Correlations')
    plt.show()

    # 5. 打印每个主题的关键词
    print("\nTop words in each topic:")
    for i, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[:-n_words-1:-1]
        top_words = [feature_names[idx] for idx in top_words_idx]
        print(f"\nTopic {i+1}:")
        print(", ".join(top_words))

# 使用这个函数来可视化LDA结果
visualize_lda_results(lda_model, count_vectorizer, lda_output)
```
# 情感分析

数据准备和增强预处理



```js
import pandas as pd
import numpy as np
import re
import spacy
from spacy import displacy
from tqdm import tqdm
tqdm.pandas()

# 加载数据和模型
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
df = pd.read_csv("preprocessed_news.csv")

# 增强的情感分析预处理
def enhanced_sentiment_preprocess(text):
    """保留情感相关词汇和上下文"""
    # 清理文本
    text = re.sub(r"http\S+|www\S+|https\S+", "", str(text).lower())
    text = re.sub(r"[^a-zA-Z\s'’]", "", text)

    # 处理否定和强调
    text = re.sub(r"\b(not|never|no)\b", " not_", text)
    text = re.sub(r"\b(very|really|extremely)\b", " very_", text)

    # 提取情感相关词汇
    doc = nlp(text)
    tokens = []
    for token in doc:
        # 保留形容词、副词、动词、否定词、情感名词
        if (token.pos_ in {"ADJ", "ADV", "VERB", "NOUN"}) or \
           (token.dep_ == "neg") or \
           (token.text.startswith(("not_", "very_"))):
            tokens.append(token.lemma_)

    return " ".join(tokens)

# 应用预处理
df["sentiment_text"] = df["text"].progress_apply(enhanced_sentiment_preprocess)
```

```js
100%|██████████| 892/892 [00:03<00:00, 229.00it/s]
```


多模型情感分析集成

```js
# 首先确保安装正确版本的库

!pip install pandas vaderSentiment textblob transformers tqdm

# 然后导入库
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline
from tqdm import tqdm
import numpy as np  # 确保这是1.24.0版本

print(f"使用的NumPy版本: {np.__version__}")  # 应该显示1.24.x

tqdm.pandas()

# 初始化各分析模型
vader = SentimentIntensityAnalyzer()

# 安全地初始化transformer模型
try:
    sentiment_pipeline = pipeline("sentiment-analysis",
                                model="distilbert-base-uncased-finetuned-sst-2-english")
except Exception as e:
    print(f"无法加载transformer模型: {e}")
    sentiment_pipeline = None

def ensemble_sentiment_analysis(text):
    """集成多种情感分析方法"""
    # 处理缺失值
    if pd.isna(text):
        text = ""

    # VADER分析
    vader_scores = vader.polarity_scores(text)

    # TextBlob分析
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    # Transformer模型分析
    transformer_score = 0
    if sentiment_pipeline is not None:
        try:
            transformer_result = sentiment_pipeline(text[:512])[0]  # 限制长度
            transformer_score = (1 if transformer_result["label"] == "POSITIVE" else -1) * transformer_result["score"]
        except Exception as e:
            print(f"Transformer分析出错: {e}")

    # 加权综合得分
    combined_score = (vader_scores["compound"] * 0.4 +
                     polarity * 0.3 +
                     transformer_score * 0.3)

    # 生成详细报告
    return {
        "vader_compound": vader_scores["compound"],
        "textblob_polarity": polarity,
        "transformer_score": transformer_score,
        "combined_score": combined_score,
        "sentiment_label": "positive" if combined_score > 0.1 else
                          "negative" if combined_score < -0.1 else
                          "neutral"
    }

# 应用情感分析
print("正在进行情感分析...")
sentiment_results = df["text"].progress_apply(ensemble_sentiment_analysis)
sentiment_df = pd.DataFrame(list(sentiment_results))
df = pd.concat([df, sentiment_df], axis=1)

print(df)
```

执行结果


```js
正在进行情感分析...
100%|██████████| 892/892 [01:21<00:00, 10.89it/s]                                                  text        category  \
0    funniest tweet cat dog week sept dog dont unde...          COMEDY   
1    funniest tweet parent week sept accidentally p...       PARENTING   
2    maury will shortstop dodger maury will helped ...          SPORTS   
3    golden globe returning nbc january year past m...   ENTERTAINMENT   
4    biden say force defend taiwan china invaded pr...        POLITICS   
..                                                 ...             ...   
887  larry headline crystal nfl legend cruise isnt ...          TRAVEL   
888  really shouldnt eat gluten group expert seven ...        WELLNESS   
889  cant wait taking action alzheimers disease can...        WELLNESS   
890  catching sucking jill taking year retail exper...  STYLE & BEAUTY   
891  tip living life action based fear going give l...        WELLNESS   

                                        sentiment_text  vader_compound  \
0                 cat dog week sept dog understand eat          0.5574   
1    tweet parent week sept accidentally put grownu...          0.3400   
2    maury shortstop dodger maury help win world se...          0.0516   
3    return year past month hollywood effectively b...          0.0516   
4                            invade issue tension rise         -0.3182   
..                                                 ...             ...   
887  first rodeo though do cruise take look celebs ...          0.0000   
888  very _ eat gluten group expert country propose...         -0.4019   
889  wait take action alzheimer disease wait age bo...          0.7506   
890  catch suck jill take year retail experience ve...          0.0000   
891  tip living life action base fear go give long ...         -0.8834   

     textblob_polarity  transformer_score  combined_score sentiment_label  
0             0.000000           0.988167        0.519410        positive  
1             0.000000          -0.982850       -0.158855        negative  
2             0.800000           0.963742        0.549763        positive  
3             0.120833          -0.998164       -0.242559        negative  
4             0.000000           0.882423        0.137447        positive  
..                 ...                ...             ...             ...  
887           0.250000          -0.662286       -0.123686        negative  
888           0.200000          -0.998155       -0.400207        negative  
889           0.133333          -0.864603        0.080859         neutral  
890           0.350000           0.999054        0.404716        positive  
891           0.016667          -0.933136       -0.628301        negative  

[892 rows x 8 columns]
```

3.实体与方面级情感分析

```js
import spacy
from tqdm import tqdm
tqdm.pandas()

# 加载英文模型（仅启用NER需要的组件）
nlp_ner = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])

def extract_entities_with_sentiment(row):
    """
    优化版的实体情感关联提取
    输入: DataFrame的一行(包含text和情感分数)
    输出: 实体及其情感信息的字典
    """
    text = row["text"]
    doc = nlp_ner(text)
    entities = {}
    
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT"]:
            # 获取实体上下文窗口
            window_size = 10  # 前后各取10个词作为上下文
            start = max(0, ent.start - window_size)
            end = min(len(doc), ent.end + window_size)
            context = doc[start:end].text
            
            # 使用已有情感分数(优化性能)
            entities[ent.text] = {
                "type": ent.label_,
                "sentiment": row["combined_score"],  # 使用整句情感分数
                "label": row["sentiment_label"],
                "context": context,
                "entity_text": ent.text
            }
    
    return entities

# 测试示例
sample_row = df.iloc[4]  # 取第5条数据
print("示例文本:", sample_row["text"])
print("提取结果:")
print(extract_entities_with_sentiment(sample_row))
```
结果
```js
示例文本: biden say force defend taiwan china invaded president issue vow tension china rise
提取结果:
{'taiwan': {'type': 'GPE', 'sentiment': np.float64(0.13744676992416383), 'label': 'positive', 'context': 'biden say force defend taiwan china invaded president issue vow tension china rise', 'entity_text': 'taiwan'}, 'china': {'type': 'GPE', 'sentiment': np.float64(0.13744676992416383), 'label': 'positive', 'context': 'say force defend taiwan china invaded president issue vow tension china rise', 'entity_text': 'china'}}
```

```js
# 批量处理函数（优化内存使用）
def batch_extract_entities(df, batch_size=1000):
    """
    分批处理DataFrame，避免内存不足
    """
    results = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        batch_results = batch.apply(extract_entities_with_sentiment, axis=1)
        results.extend(batch_results)
    return results

# 应用到整个数据集
print("正在提取实体及其情感...")
df["entity_sentiments"] = batch_extract_entities(df)
```
执行结果

```js
正在提取实体及其情感...
100%|██████████| 1/1 [00:08<00:00,  8.67s/it]
```

情感分析可视化

```js
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud

# 设置 seaborn 绘图风格
sns.set_style("whitegrid")  # 设置 seaborn 的风格，这里使用 whitegrid 风格，你也可以根据需要选择其他风格
sns.set_palette("husl")



# 1. 整体情感分布
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=df, x='sentiment_label', 
                 order=['positive', 'neutral', 'negative'],
                 palette={'positive': '#4caf50', 'neutral': '#9e9e9e', 'negative': '#f44336'})
plt.title('News sentiment label distribution', fontsize=14, pad=20)
plt.xlabel('Sentiment classification', fontsize=12)
plt.ylabel('quantity', fontsize=12)

# 添加数值标签
for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}', 
               (p.get_x() + p.get_width() / 2., p.get_height()),
               ha='center', va='center', 
               xytext=(0, 5), 
               textcoords='offset points')

plt.tight_layout()
plt.show()
```

![image](https://github.com/user-attachments/assets/094c38d8-c33b-4c77-a49c-36e8f7d45a7c)

```js
# 2. 多模型情感分数分布对比
plt.figure(figsize=(14, 8))

# 创建子图网格
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# VADER分数分布
sns.histplot(data=df, x='vader_compound', bins=30, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('VADER sentiment score distribution')
axes[0, 0].axvline(0, color='k', linestyle='--')

# TextBlob分数分布
sns.histplot(data=df, x='textblob_polarity', bins=30, kde=True, ax=axes[0, 1])
axes[0, 1].set_title('TextBlobsentiment score distribution')
axes[0, 1].axvline(0, color='k', linestyle='--')

# Transformer分数分布
sns.histplot(data=df, x='transformer_score', bins=30, kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Transformersentiment score distribution')
axes[1, 0].axvline(0, color='k', linestyle='--')

# 综合分数分布
sns.histplot(data=df, x='combined_score', bins=30, kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Comprehensive sentiment score distribution')
axes[1, 1].axvline(0, color='k', linestyle='--')

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/6937d175-bd08-4fca-bb72-7c0e7b8fd595)

```js
# 3. 各类别情感分布
plt.figure(figsize=(14, 8))

# 按类别分组计算平均情感分数
category_sentiment = df.groupby('category')['combined_score'].agg(['mean', 'count'])
category_sentiment = category_sentiment[category_sentiment['count'] > 50]  # 过滤样本量少的类别

# 绘制条形图
ax = sns.barplot(data=category_sentiment.reset_index(), 
                x='mean', y='category',
                palette='coolwarm')
plt.axvline(0, color='k', linestyle='--')
plt.title('Average sentiment score of each news category', fontsize=14, pad=20)
plt.xlabel('average sentiment score', fontsize=12)
plt.ylabel('News categories', fontsize=12)

# 添加数值标签
for i, (mean, count) in enumerate(zip(category_sentiment['mean'], category_sentiment['count'])):
    ax.text(mean, i, f'{mean:.2f}\n(n={count})', 
            va='center', ha='left' if mean < 0 else 'right',
            color='white' if abs(mean) > 0.2 else 'black')

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/b24cc0e9-6fcb-4c03-b70f-aa58c131a376)

```js
# 4. 情感词云对比
def generate_sentiment_wordcloud(sentiment_type='positive', colormap='Greens'):
    """生成情感词云"""
    subset = df[df['sentiment_label'] == sentiment_type]
    text = ' '.join(subset['sentiment_text'])
    
    # 过滤停用词
    stopwords = set(['say', 'said', 'will', 'one', 'year', 'new'])
    
    wordcloud = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap=colormap,
        max_words=100,
        stopwords=stopwords,
        contour_width=3,
        contour_color='steelblue'
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud)
    plt.title(f'{sentiment_type.capitalize()}Emotion word cloud', fontsize=14, pad=20)
    plt.axis("off")
    plt.show()

# 生成正面和负面词云
generate_sentiment_wordcloud('positive', 'Greens')
generate_sentiment_wordcloud('negative', 'Reds')
```
![image](https://github.com/user-attachments/assets/391ca59c-67d6-4851-bbea-ddb87c429fea)

![image](https://github.com/user-attachments/assets/e1ad3297-f85d-4d1d-9d81-45e135308c94)

实体情感分析

```js
# 5. 实体情感分析
from collections import defaultdict

# 提取所有实体及其情感分数
entity_sentiments = defaultdict(list)
for entities in df['entity_sentiments']:
    for ent, data in entities.items():
        entity_sentiments[ent].append(data['sentiment'])

# 计算每个实体的平均情感
entity_avg_sentiment = {
    ent: np.mean(scores) 
    for ent, scores in entity_sentiments.items() 
    if len(scores) >= 5  # 只考虑出现5次以上的实体
}

# 转换为DataFrame
entity_df = pd.DataFrame({
    'entity': list(entity_avg_sentiment.keys()),
    'avg_sentiment': list(entity_avg_sentiment.values()),
    'count': [len(entity_sentiments[ent]) for ent in entity_avg_sentiment.keys()]
})

# 筛选最具代表性的实体
top_entities = entity_df.nlargest(10, 'count')

# 可视化
plt.figure(figsize=(14, 8))
ax = sns.barplot(data=top_entities, x='avg_sentiment', y='entity',
                palette='coolwarm')
plt.axvline(0, color='k', linestyle='--')
plt.title('High-frequency entity sentiment analysis (occurrence ≥ 5 times', fontsize=14, pad=20)
plt.xlabel('average sentiment score', fontsize=12)
plt.ylabel('Entity name', fontsize=12)

# 添加数值标签
for p in ax.patches:
    width = p.get_width()
    ax.annotate(f'{width:.2f}',
                (width, p.get_y() + p.get_height() / 2.),
                ha='left' if width > 0 else 'right',
                va='center',
                xytext=(5 if width > 0 else -5, 0),
                textcoords='offset points')

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/822b6ea2-f09b-4430-b3b8-6a62ea9cf1a4)

```js
from ipywidgets import interact, widgets
import plotly.express as px

# 6.1 交互式情感探索
@interact
def explore_sentiment(category=widgets.Dropdown(
    options=sorted(df['category'].unique())),
    score_type=widgets.Dropdown(
    options=['combined_score', 'vader_compound', 'textblob_polarity'])):
    """交互式探索不同类别的情感分布"""
    subset = df[df['category'] == category]
    
    fig = px.histogram(subset, x=score_type, 
                      nbins=30,
                      title=f'{category} category {score_type} score_type',
                      color_discrete_sequence=['#636EFA'])
    
    fig.update_layout(
        xaxis_title='sentiment score',
        yaxis_title='quantity',
        bargap=0.1
    )
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.show()

# 6.2 交互式实体探索
if len(entity_df) > 0:
    @interact
    def explore_entity(min_count=widgets.IntSlider(
        min=1, max=entity_df['count'].max(), value=5)):
        """交互式探索实体情感"""
        filtered = entity_df[entity_df['count'] >= min_count]
        
        fig = px.scatter(filtered, x='avg_sentiment', y='entity',
                        size='count', color='avg_sentiment',
                        color_continuous_scale='RdYlGn',
                        title=f'Entity sentiment analysis (occurrence ≥{min_count} Second - rate)')
        
        fig.update_layout(
            xaxis_title='average sentiment score',
            yaxis_title='Entity name',
            coloraxis_colorbar=dict(title="sentiment score")
        )
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        fig.show()
```
![image](https://github.com/user-attachments/assets/22b2bd7e-b3b0-4602-91b0-fcc3444aeebd)



![image](https://github.com/user-attachments/assets/26682df9-1b40-4582-837f-73a615be3233)

```js
# 7. 生成高级分析报告
def generate_sentiment_report():
    """生成综合情感分析报告"""
    report = {
        'overall_sentiment': {
            'positive': len(df[df['sentiment_label'] == 'positive']),
            'neutral': len(df[df['sentiment_label'] == 'neutral']),
            'negative': len(df[df['sentiment_label'] == 'negative'])
        },
        'top_positive_category': df.groupby('category')['combined_score'].mean().idxmax(),
        'top_negative_category': df.groupby('category')['combined_score'].mean().idxmin(),
        'most_positive_entity': max(entity_avg_sentiment.items(), key=lambda x: x[1])[0] if entity_avg_sentiment else "N/A",
        'most_negative_entity': min(entity_avg_sentiment.items(), key=lambda x: x[1])[0] if entity_avg_sentiment else "N/A"
    }
    
    # 打印报告
    print("="*50)
    print("新闻情感分析综合报告".center(40))
    print("="*50)
    print(f"\n整体情感分布:")
    print(f"- 正面新闻: {report['overall_sentiment']['positive']}篇")
    print(f"- 中性新闻: {report['overall_sentiment']['neutral']}篇")
    print(f"- 负面新闻: {report['overall_sentiment']['negative']}篇")
    
    print(f"\n最具正面情感的新闻类别: {report['top_positive_category']}")
    print(f"最具负面情感的新闻类别: {report['top_negative_category']}")
    
    if entity_avg_sentiment:
        print(f"\n最正面关联实体: {report['most_positive_entity']}")
        print(f"最负面关联实体: {report['most_negative_entity']}")
    
    print("\n" + "="*50)

# 生成报告
generate_sentiment_report()
```


结果

```js
==================================================
新闻情感分析综合报告News Sentiment Analysis Comprehensive Report
==================================================

整体情感分布Overall sentiment distribution:
- 正面新闻Positive News: 317篇
- 中性新闻Neutral News: 94篇
- 负面新闻Negative News: 481篇

最具正面情感的新闻类别News categories with the most positive sentiment: FOOD & DRINK
最具负面情感的新闻类别News categories with the most negative sentiment: POLITICS

最正面关联实体Most positively associated entity: los angeles
最负面关联实体Most Negatively Associated Entity: texas

==================================================
```
商业应用场景实现


5.1 竞争实体对比分析


```js
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

def compare_entities(df, entity1=None, entity2=None):
    """
    比较两个实体的情感表现
    参数:
        df: 包含实体情感数据的DataFrame
        entity1: 第一个实体名称 (可选)
        entity2: 第二个实体名称 (可选)
    """
    # 检查entity_sentiments列是否存在
    if "entity_sentiments" not in df.columns:
        print("数据中缺少 'entity_sentiments' 列")
        return
    
    # 如果没有指定实体，自动选择数据中最常见的两个实体
    if entity1 is None or entity2 is None:
        entity_counts = defaultdict(int)
        for entities in df["entity_sentiments"]:
            if isinstance(entities, dict):
                for ent in entities.keys():
                    entity_counts[ent] += 1
        
        if len(entity_counts) < 2:
            print("数据中可比较的实体不足 (至少需要2个不同实体)")
            return
        
        # 选择出现频率最高的两个不同实体
        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        entity1, entity2 = top_entities[0][0], top_entities[1][0]
        print(f"自动选择实体: {entity1} vs {entity2}")
    
    # 收集实体情感数据
    entity1_data = []
    entity2_data = []
    
    for _, row in df.iterrows():
        entities = row["entity_sentiments"]
        if isinstance(entities, dict):
            if entity1 in entities:
                entity1_data.append(entities[entity1].get("sentiment", 0))
            if entity2 in entities:
                entity2_data.append(entities[entity2].get("sentiment", 0))
    
    # 检查是否有足够数据
    if not entity1_data or not entity2_data:
        print(f"没有足够数据进行比较 (实体1: {len(entity1_data)}条, 实体2: {len(entity2_data)}条)")
        return
    
    # 可视化比较
    plt.figure(figsize=(14, 6))
    
    # 子图1: 情感分布直方图
    plt.subplot(1, 2, 1)
    sns.histplot(entity1_data, bins=15, color="blue", kde=True, alpha=0.6, label=entity1)
    sns.histplot(entity2_data, bins=15, color="orange", kde=True, alpha=0.6, label=entity2)
    plt.axvline(np.mean(entity1_data), color="blue", linestyle="--")
    plt.axvline(np.mean(entity2_data), color="orange", linestyle="--")
    plt.title(f"{entity1} vs {entity2} 情感分布比较")
    plt.xlabel("情感分数")
    plt.legend()
    
    # 子图2: 箱线图比较
    plt.subplot(1, 2, 2)
    plot_data = pd.DataFrame({
        "Entity": [entity1]*len(entity1_data) + [entity2]*len(entity2_data),
        "Sentiment": entity1_data + entity2_data
    })
    sns.boxplot(data=plot_data, x="Entity", y="Sentiment", 
                palette={"blue", "orange"}, width=0.4)
    plt.title("情感分数箱线图比较")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print(f"\n{entity1} 情感分析结果:")
    print(f"- 平均分: {np.mean(entity1_data):.2f}")
    print(f"- 标准差: {np.std(entity1_data):.2f}")
    print(f"- 数据点: {len(entity1_data)}条")
    
    print(f"\n{entity2} 情感分析结果:")
    print(f"- 平均分: {np.mean(entity2_data):.2f}")
    print(f"- 标准差: {np.std(entity2_data):.2f}")
    print(f"- 数据点: {len(entity2_data)}条")
    
    # 执行t检验 (如果数据足够)
    if len(entity1_data) > 1 and len(entity2_data) > 1:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(entity1_data, entity2_data)
        print(f"\n独立样本t检验结果:")
        print(f"- t统计量: {t_stat:.2f}")
        print(f"- p值: {p_value:.4f}")
        if p_value < 0.05:
            print("-> 差异显著 (p < 0.05)")
        else:
            print("-> 差异不显著")

# 使用示例
print("="*50)
print("实体情感比较分析")
print("="*50)

# 方式1: 自动选择最常见的两个实体比较
compare_entities(df)

# 方式2: 手动指定实体比较 (根据您的实际数据中的实体名称)
# compare_entities(df, entity1="funniest tweet parent", entity2="grownup toothpaste")
```

结果

![image](https://github.com/user-attachments/assets/f19301fb-f0fe-4185-9bf3-c1dd8597907c)


```js

gop Sentiment analysis results:
- average score: -0.20
- standard deviation: 0.48
- data points: 32条

donald trump Sentiment analysis results:
- average score: -0.12
- standard deviation: 0.37
- data points: 18条

 Independent samples t-test results:
- t-statistic: -0.62
- p value: 0.5413
-> The difference is not significant
```

5.2 舆情预警系统

```js
def sentiment_alert_system(df, threshold=-0.2):
    """
    负面舆情预警系统
    参数:
        df: 包含情感分析结果的DataFrame
        threshold: 负面情感阈值 (默认-0.2)
    """
    # 检查必要的列是否存在
    required_columns = ['entity_sentiments']
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        print(f"⚠️ Missing required columns in data: {missing_cols}")
        return

    # 尝试不同的情感分数列 (按优先级检查)
    score_columns = ['combined_score', 'vader_compound', 'textblob_polarity']
    score_col = next((col for col in score_columns if col in df.columns), None)

    if score_col is None:
        print("⚠️ No available sentiment score columns found in data")
        return

    print(f"Using '{score_col}' as sentiment score reference")

    # 按类别检测负面情绪
    negative_news = df[df[score_col] < threshold]

    if len(negative_news) > 0:
        print(f"\n⚠️ Detected {len(negative_news)} negative news items (score < {threshold})")

        # 统计高频负面实体
        negative_entities = {}
        for entities in negative_news["entity_sentiments"]:
            # 确保entity_sentiments是字典格式
            if isinstance(entities, str):
                try:
                    entities = eval(entities)  # 将字符串转换为字典
                except:
                    continue

            if isinstance(entities, dict):
                for ent, data in entities.items():
                    if isinstance(data, dict) and 'sentiment' in data:
                        if data['sentiment'] < threshold:
                            negative_entities[ent] = negative_entities.get(ent, 0) + 1

        # 展示Top负面实体
        if negative_entities:
            top_negative = sorted(negative_entities.items(),
                                 key=lambda x: x[1], reverse=True)[:5]
            print("\n🔥 Top negative associated entities:")
            for ent, count in top_negative:
                print(f"- {ent}: {count} times")
        else:
            print("\nNo significant negative associated entities found")

        # 展示代表性负面新闻
        print("\n📌 Representative negative news samples:")
        sample_size = min(3, len(negative_news))
        for idx, row in negative_news.nsmallest(sample_size, score_col).iterrows():
            print(f"\n[{idx}] Sentiment score: {row[score_col]:.2f}")

            # 尝试获取标题或文本片段
            if 'headline' in df.columns:
                print(f"Headline: {row['headline']}")
            elif 'text' in df.columns:
                preview = row['text'][:50] + "..." if len(row['text']) > 50 else row['text']
                print(f"Content: {preview}")

            # 显示关联实体
            entities = row['entity_sentiments']
            if isinstance(entities, str):
                try:
                    entities = eval(entities)
                except:
                    entities = {}

            if isinstance(entities, dict):
                print("Key entities:", ", ".join(entities.keys()))
    else:
        print("\n✅ No significant negative sentiment detected")

# 使用示例
print("="*50)
print("Negative Sentiment Alert System Started")
print("="*50)
sentiment_alert_system(df)  # 传入您的DataFrame
```


jieguo
```js
==================================================
Negative Sentiment Alert System Started
==================================================
Using 'combined_score' as sentiment score reference

⚠️ Detected 402 negative news items (score < -0.2)

🔥 Top negative associated entities:
- gop: 22 times
- texas: 12 times
- senate: 11 times
- white house: 9 times
- donald trump: 9 times

📌 Representative negative news samples:

[70] Sentiment score: -0.95
Content: jared kushner blast nasty troll chrissy teigen att...
Key entities: jared kushner, troll

[539] Sentiment score: -0.91
Content: cory booker explains went bat ketanji brown jackso...
Key entities: cory booker, ketanji brown jackson, dick durbin, gop

[797] Sentiment score: -0.90
Content: exnfl player say violently arrested cop mistook ph...
Key entities: desmond marrow
```
技术挑战解决方案
6.1 高级讽刺检测

```js
from transformers import pipeline

# 加载讽刺检测模型
irony_pipeline = pipeline("text-classification",
                         model="cardiffnlp/twitter-roberta-base-irony")

def detect_irony(text):
    """使用Transformer模型检测讽刺"""
    try:
        result = irony_pipeline(text[:512])[0]  # 限制长度
        return result["label"] == "irony"
    except:
        return False

# 应用到疑似讽刺文本
potential_irony = df[(df["vader_compound"] > 0) & (df["textblob_polarity"] < 0)]
if len(potential_irony) > 0:
    potential_irony["is_ironic"] = potential_irony["text"].progress_apply(detect_irony)
    print(f"Detected {potential_irony['is_ironic'].sum()} ironic news items")
```
结果

```js
config.json: 100%
 705/705 [00:00<00:00, 24.3kB/s]
pytorch_model.bin: 100%
 499M/499M [00:06<00:00, 130MB/s]
model.safetensors: 100%
 499M/499M [00:08<00:00, 72.0MB/s]
vocab.json: 100%
 899k/899k [00:00<00:00, 5.03MB/s]
merges.txt: 100%
 456k/456k [00:00<00:00, 6.88MB/s]
special_tokens_map.json: 100%
 150/150 [00:00<00:00, 4.83kB/s]
Device set to use cpu

  0%|          | 0/63 [00:00<?, ?it/s]
  3%|▎         | 2/63 [00:00<00:10,  5.70it/s]
  5%|▍         | 3/63 [00:00<00:12,  4.85it/s]
  6%|▋         | 4/63 [00:00<00:13,  4.32it/s]
  8%|▊         | 5/63 [00:01<00:13,  4.19it/s]
 10%|▉         | 6/63 [00:01<00:13,  4.22it/s]
 11%|█         | 7/63 [00:01<00:12,  4.33it/s]
 13%|█▎        | 8/63 [00:01<00:12,  4.28it/s]
 14%|█▍        | 9/63 [00:02<00:13,  3.98it/s]
 16%|█▌        | 10/63 [00:02<00:14,  3.69it/s]
 17%|█▋        | 11/63 [00:02<00:16,  3.19it/s]
 19%|█▉        | 12/63 [00:03<00:22,  2.22it/s]
 21%|██        | 13/63 [00:04<00:35,  1.41it/s]
 22%|██▏       | 14/63 [00:05<00:39,  1.23it/s]
 24%|██▍       | 15/63 [00:06<00:41,  1.16it/s]
 25%|██▌       | 16/63 [00:07<00:37,  1.25it/s]
 27%|██▋       | 17/63 [00:07<00:29,  1.56it/s]
 29%|██▊       | 18/63 [00:08<00:22,  1.96it/s]
 30%|███       | 19/63 [00:08<00:19,  2.21it/s]
 32%|███▏      | 20/63 [00:08<00:18,  2.28it/s]
 33%|███▎      | 21/63 [00:09<00:17,  2.39it/s]
 35%|███▍      | 22/63 [00:09<00:14,  2.83it/s]
 37%|███▋      | 23/63 [00:09<00:14,  2.80it/s]
 38%|███▊      | 24/63 [00:10<00:15,  2.52it/s]
 40%|███▉      | 25/63 [00:10<00:14,  2.65it/s]
 41%|████▏     | 26/63 [00:10<00:14,  2.55it/s]
 43%|████▎     | 27/63 [00:11<00:15,  2.33it/s]
 44%|████▍     | 28/63 [00:11<00:14,  2.37it/s]
 46%|████▌     | 29/63 [00:12<00:13,  2.54it/s]
 48%|████▊     | 30/63 [00:12<00:12,  2.75it/s]
 49%|████▉     | 31/63 [00:12<00:12,  2.64it/s]
 51%|█████     | 32/63 [00:13<00:11,  2.82it/s]
 52%|█████▏    | 33/63 [00:13<00:10,  2.96it/s]
 54%|█████▍    | 34/63 [00:13<00:09,  2.97it/s]
 56%|█████▌    | 35/63 [00:14<00:09,  2.86it/s]
 57%|█████▋    | 36/63 [00:14<00:10,  2.59it/s]
 59%|█████▊    | 37/63 [00:15<00:10,  2.55it/s]
 60%|██████    | 38/63 [00:16<00:14,  1.78it/s]
 62%|██████▏   | 39/63 [00:16<00:15,  1.57it/s]
 63%|██████▎   | 40/63 [00:18<00:18,  1.24it/s]
 65%|██████▌   | 41/63 [00:18<00:16,  1.34it/s]
 67%|██████▋   | 42/63 [00:19<00:14,  1.46it/s]
 68%|██████▊   | 43/63 [00:19<00:11,  1.72it/s]
 70%|██████▉   | 44/63 [00:19<00:09,  1.98it/s]
 71%|███████▏  | 45/63 [00:20<00:08,  2.07it/s]
 73%|███████▎  | 46/63 [00:20<00:08,  2.12it/s]
 75%|███████▍  | 47/63 [00:21<00:07,  2.26it/s]
 76%|███████▌  | 48/63 [00:21<00:06,  2.32it/s]
 78%|███████▊  | 49/63 [00:21<00:05,  2.45it/s]
 79%|███████▉  | 50/63 [00:22<00:04,  2.64it/s]
 81%|████████  | 51/63 [00:22<00:04,  2.78it/s]
 83%|████████▎ | 52/63 [00:22<00:03,  3.00it/s]
 84%|████████▍ | 53/63 [00:22<00:02,  3.51it/s]
 86%|████████▌ | 54/63 [00:23<00:02,  3.93it/s]
 87%|████████▋ | 55/63 [00:23<00:01,  4.76it/s]
 89%|████████▉ | 56/63 [00:23<00:01,  4.93it/s]
 90%|█████████ | 57/63 [00:23<00:01,  4.71it/s]
 92%|█████████▏| 58/63 [00:23<00:01,  4.74it/s]
 94%|█████████▎| 59/63 [00:24<00:00,  4.70it/s]
 95%|█████████▌| 60/63 [00:24<00:00,  4.79it/s]
 97%|█████████▋| 61/63 [00:24<00:00,  4.63it/s]
 98%|█████████▊| 62/63 [00:24<00:00,  4.69it/s]
100%|██████████| 63/63 [00:25<00:00,  2.50it/s]Detected 36 ironic news items

<ipython-input-25-9f0eea376282>:18: SettingWithCopyWarning:


A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
```

![image](https://github.com/user-attachments/assets/94a59fe6-82e4-48ed-b137-c0d0b385ba59)


6.2 动态领域自适应


```js
def dynamic_domain_adaptation(text, category):
    """动态领域适应情感分析"""
    # 领域特定调整规则
    domain_rules = {
        "POLITICS": {
            "boost_words": ["reform", "progress", "bipartisan"],
            "penalty_words": ["scandal", "corruption", "controversy"]
        },
        "TECH": {
            "boost_words": ["innovative", "sleek", "user-friendly"],
            "penalty_words": ["buggy", "outdated", "overpriced"]
        }
    }

    # 获取基础分数
    base_score = ensemble_sentiment_analysis(text)["combined_score"]

    # 应用领域调整
    if category in domain_rules:
        boost = sum(1 for word in domain_rules[category]["boost_words"]
                if word in text.lower()) * 0.05
        penalty = sum(1 for word in domain_rules[category]["penalty_words"]
                     if word in text.lower()) * 0.05
        adjusted_score = base_score + boost - penalty
        return max(-1, min(1, adjusted_score))  # 保持在[-1,1]范围内

    return base_score

# 应用领域自适应
df["domain_adjusted_score"] = df.progress_apply(
    lambda row: dynamic_domain_adaptation(row["text"], row["category"]), axis=1)
```


结果

系统集成与部署

```js
import pandas as pd
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

# 下载 vader_lexicon 资源
nltk.download('vader_lexicon')

# 假设这些函数已经定义
def enhanced_sentiment_preprocess(text):
    return text

def ensemble_sentiment_analysis(text):
    return "positive"

def extract_entities_with_sentiment(text):
    """提取实体及其情感信息"""
    nlp_ner = spacy.load("en_core_web_sm")
    doc = nlp_ner(text)
    entities = {}
    for ent in doc.ents:
        entities[ent.text] = {
            "label": ent.label_,
            # 这里简单假设情感都是中性，实际需要根据情况实现情感分析
            "sentiment": "neutral"
        }
    return entities

def detect_irony(text):
    return False


class NewsSentimentAnalyzer:
    def __init__(self, data_path):
        """初始化分析系统"""
        self.df = self._load_data(data_path)
        self._load_models()

    def _load_data(self, path):
        """加载数据"""
        df = pd.read_csv(path)
        return df

    def _load_models(self):
        """加载所需模型"""
        print("Loading NLP models...")
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.nlp_ner = spacy.load("en_core_web_sm")
        self.vader = SentimentIntensityAnalyzer()
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.irony_pipeline = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-irony"
        )

    def analyze(self, text):
        """分析单条新闻"""
        # 预处理
        clean_text = enhanced_sentiment_preprocess(text)

        # 情感分析
        sentiment = ensemble_sentiment_analysis(text)

        # 实体提取
        entities = extract_entities_with_sentiment(text)

        # 讽刺检测
        is_ironic = detect_irony(text)

        return {
            "text": text,
            "clean_text": clean_text,
            "sentiment": sentiment,
            "entities": entities,
            "is_ironic": is_ironic
        }

    def batch_analyze(self, texts):
        """批量分析新闻"""
        return [self.analyze(text) for text in texts]

    def generate_report(self):
        """生成分析报告"""
        # 实现报告生成逻辑...
        pass

# 使用示例
analyzer = NewsSentimentAnalyzer("preprocessed_news.csv")
df = analyzer.df
sample_news = df.iloc[0]["text"]
print(analyzer.analyze(sample_news))
```
结果

```js
[nltk_data] Downloading package vader_lexicon to /root/nltk_data...
Loading NLP models...
Device set to use cpu
Device set to use cpu
{'text': 'funniest tweet cat dog week sept dog dont understand eaten', 'clean_text': 'funniest tweet cat dog week sept dog dont understand eaten', 'sentiment': 'positive', 'entities': {'week sept': {'label': 'DATE', 'sentiment': 'neutral'}}, 'is_ironic': False}
```

```js
import pandas as pd
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

# 下载 vader_lexicon 资源
nltk.download('vader_lexicon')

# 重新加载 spacy 模型，不禁用解析器
nlp = spacy.load("en_core_web_sm")


def enhanced_sentiment_preprocess(text):
    return text


def ensemble_sentiment_analysis(text):
    return "positive"


def extract_entities_with_sentiment(text):
    """提取实体及其情感信息"""
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        entities[ent.text] = {
            "label": ent.label_,
            # 这里简单假设情感都是中性，实际需要根据情况实现情感分析
            "sentiment": "neutral"
        }
    return entities


def detect_irony(text):
    return False


def extract_entities_and_aspects(text):
    doc = nlp(text)
    entities = []
    aspects = []

    # 提取命名实体(产品、组织、地点等)
    for ent in doc.ents:
        if ent.label_ in ('ORG', 'PRODUCT', 'PERSON', 'GPE'):
            entities.append((ent.text, ent.label_))

    # 提取方面(名词短语)
    for chunk in doc.noun_chunks:
        # 过滤通用名词
        if chunk.root.pos_ == 'NOUN' and len(chunk.text) > 3:
            aspects.append(chunk.text)

    return list(set(entities)), list(set(aspects))


class NewsSentimentAnalyzer:
    def __init__(self, data_path):
        """初始化分析系统"""
        self.df = self._load_data(data_path)
        self._load_models()

    def _load_data(self, path):
        """加载数据"""
        df = pd.read_csv(path)
        return df

    def _load_models(self):
        """加载所需模型"""
        print("Loading NLP models...")
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp_ner = spacy.load("en_core_web_sm")
        self.vader = SentimentIntensityAnalyzer()
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.irony_pipeline = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-irony"
        )

    def analyze(self, text):
        """分析单条新闻"""
        # 预处理
        clean_text = enhanced_sentiment_preprocess(text)

        # 情感分析
        sentiment = ensemble_sentiment_analysis(text)

        # 实体提取
        entities = extract_entities_with_sentiment(text)

        # 讽刺检测
        is_ironic = detect_irony(text)

        return {
            "text": text,
            "clean_text": clean_text,
            "sentiment": sentiment,
            "entities": entities,
            "is_ironic": is_ironic
        }

    def batch_analyze(self, texts):
        """批量分析新闻"""
        return [self.analyze(text) for text in texts]

    def generate_report(self):
        """生成分析报告"""
        # 实现报告生成逻辑...
        pass


# 使用示例
analyzer = NewsSentimentAnalyzer("preprocessed_news.csv")
df = analyzer.df
sample_text = "The new iPhone's battery life is disappointing compared to Samsung's latest model."
entities, aspects = extract_entities_and_aspects(sample_text)
print("Entities:", entities)
print("Aspects:", aspects)

# 应用到整个数据集(抽样部分数据)
df[['entities', 'aspects']] = df['text'].apply(
    lambda x: pd.Series(extract_entities_and_aspects(x)))
```

结果

```js
[nltk_data] Downloading package vader_lexicon to /root/nltk_data...
[nltk_data]   Package vader_lexicon is already up-to-date!
Loading NLP models...
Device set to use cpu
Device set to use cpu
Entities: [('iPhone', 'ORG'), ('Samsung', 'ORG')]
Aspects: ["Samsung's latest model", "The new iPhone's battery life"]
```
# 关联规则挖掘


```js
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

def prepare_transaction_data(df):
    """
    按课程标准准备事务数据
    修改点：
    1. 移除不必要的返回项
    2. 简化特征提取逻辑
    3. 确保输出格式与课程示例完全一致
    """
    transactions = []
    for _, row in df.iterrows():
        itemset = []
        
        # 必须包含情感标签（课程中的Y变量）
        itemset.append(f"sentiment={row['sentiment_label']}")
        
        # 仅保留课程要求的特征类型
        if 'category' in df.columns:
            itemset.append(f"category={row['category']}")
        
        if 'entity_sentiments' in df.columns and isinstance(row['entity_sentiments'], dict):
            itemset.extend([f"entity={ent}" for ent in row['entity_sentiments'].keys()])
        
        transactions.append(itemset)
    
    # 严格按课程示例进行one-hot编码
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)

# 生成事务数据
trans_df = prepare_transaction_data(df)
print("事务数据示例（前5项）：")
print(trans_df.iloc[0, :5])  # 显示第一行的前5个特征
```

结果

事务数据示例（前5项）：

category=BLACK VOICES     False

category=BUSINESS         False

category=COMEDY            True

category=ENTERTAINMENT    False

category=FOOD & DRINK     False

Name: 0, dtype: bool


```js
from mlxtend.frequent_patterns import apriori, association_rules

def mine_association_rules(trans_df):
    """
    按课程标准实现关联规则挖掘
    修改点：
    1. 使用课程指定的默认参数
    2. 简化规则筛选逻辑
    3. 输出格式与课程示例一致
    """
    # 使用课程推荐参数（min_support=0.1, min_threshold=0.7）
    frequent_itemsets = apriori(trans_df, min_support=0.1, use_colnames=True)
    
    # 生成规则（按课程要求使用confidence作为度量）
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    
    # 按课程示例添加length列
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    
    # 筛选有效规则（lift>1）
    meaningful_rules = rules[rules['lift'] > 1].copy()
    meaningful_rules['rule'] = meaningful_rules.apply(
        lambda row: f"{set(row['antecedents'])} → {set(row['consequents'])}", 
        axis=1
    )
    
    return frequent_itemsets, meaningful_rules

# 执行挖掘
frequent_itemsets, rules = mine_association_rules(trans_df)

# 显示结果（按课程表格格式）
print("\n频繁项集（前5个）：")
print(frequent_itemsets[['itemsets', 'support', 'length']].head())
print("\n关联规则（前5条）：")
print(rules[['rule', 'support', 'confidence', 'lift']].head())
```
结果

```js
频繁项集（前5个）：
                   itemsets   support  length
0  (category=ENTERTAINMENT)  0.227578       1
1       (category=POLITICS)  0.316143       1
2      (sentiment=negative)  0.539238       1
3       (sentiment=neutral)  0.105381       1
4      (sentiment=positive)  0.355381       1

关联规则（前5条）：
                                             rule   support  confidence  \
0  {'category=POLITICS'} → {'sentiment=negative'}  0.232063    0.734043   

      lift  
0  1.36126
```


```js
import matplotlib.pyplot as plt

def plot_rules(rules):
    """
    按课程要求绘制规则热力图
    修改点：
    1. 使用课程指定的可视化形式
    2. 简化视觉元素
    """
    if len(rules) == 0:
        print("无有效规则可可视化")
        return
    
    plt.figure(figsize=(10, 6))
    plt.scatter(
        rules['support'],
        rules['confidence'],
        s=rules['lift']*50,
        c=rules['lift'],
        cmap='viridis',
        alpha=0.6
    )
    
    # 添加课程要求的标签
    plt.colorbar(label='Lift值')
    plt.xlabel('Support', fontsize=12)
    plt.ylabel('Confidence', fontsize=12)
    plt.title('关联规则热力图（按课程规范）', fontsize=14)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

# 可视化
plot_rules(rules)
```

![image](https://github.com/user-attachments/assets/662e8a14-167b-45f8-8810-98191a2a6dd2)



```js
# 1. 数据准备
print("=== 步骤1：准备事务数据 ===")
trans_df = prepare_transaction_data(df)

# 2. 频繁项集挖掘
print("\n=== 步骤2：Apriori算法执行 ===")
frequent_itemsets, rules = mine_association_rules(trans_df)

# 3. 结果分析
print("\n=== 步骤3：规则分析 ===")
if len(rules) > 0:
    # 按课程要求保存结果
    rules.to_csv("association_rules.csv", index=False)
    print("已保存规则到association_rules.csv")
    
    # 情感相关规则分析（课程项目要求）
    sentiment_rules = rules[rules['consequents'].apply(
        lambda x: any('sentiment=' in item for item in x)
    )]
    print("\n情感相关规则（前5条）：")
    print(sentiment_rules[['rule', 'support', 'confidence', 'lift']].head())
else:
    print("未发现显著规则，建议降低min_support或min_threshold")

# 4. 可视化
print("\n=== 步骤4：可视化 ===")
plot_rules(rules)
```

结果

```js
=== 步骤1：准备事务数据 ===

=== 步骤2：Apriori算法执行 ===

=== 步骤3：规则分析 ===
已保存规则到association_rules.csv

情感相关规则（前5条）：
                                             rule   support  confidence  \
0  {'category=POLITICS'} → {'sentiment=negative'}  0.232063    0.734043   

      lift  
0  1.36126  

=== 步骤4：可视化 ===
<ipython-input-39-fa4fe257c0ca>:30: UserWarning:

Glyph 20540 (\N{CJK UNIFIED IDEOGRAPH-503C}) missing from font(s) DejaVu Sans.

/usr/local/lib/python3.11/dist-packages/IPython/core/pylabtools.py:151: UserWarning:

Glyph 20540 (\N{CJK UNIFIED IDEOGRAPH-503C}) missing from font(s) DejaVu Sans.
```
![image](https://github.com/user-attachments/assets/58334a40-3889-40ab-ab04-5e80797cfe51)

```js
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 准备关联规则挖掘所需的数据
def prepare_association_data(df):
    """
    准备关联规则挖掘的数据集
    返回:
        - 处理后的交易数据集 (用于apriori算法)
        - 原始DataFrame的扩展版本 (包含one-hot编码特征)
    """
    # 确保必要的列存在
    if 'sentiment_label' not in df.columns:
        print("请先运行情感分析!")
        return None, None
    
    # 1. 提取每行文本的特征
    transactions = []
    for _, row in df.iterrows():
        features = []
        
        # 添加情感标签
        features.append(f"sentiment={row['sentiment_label']}")
        
        # 添加新闻类别 (如果存在)
        if 'category' in df.columns:
            features.append(f"category={row['category']}")
        
        # 添加入选实体 (如果存在)
        if 'entity_sentiments' in df.columns and isinstance(row['entity_sentiments'], dict):
            for entity in row['entity_sentiments'].keys():
                features.append(f"entity={entity}")
        
        # 添加高频关键词 (从sentiment_text提取)
        if 'sentiment_text' in df.columns:
            top_words = [word for word in str(row['sentiment_text']).split() 
                        if len(word) > 3][:5]  # 取长度>3的前5个词
            for word in top_words:
                features.append(f"word={word.lower()}")
        
        transactions.append(features)
    
    # 2. 转换为one-hot编码格式
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    trans_df = pd.DataFrame(te_ary, columns=te.columns_)
    
    return transactions, trans_df

# 准备数据
transactions, trans_df = prepare_association_data(df)
print(f"生成 {len(trans_df.columns)} 个特征:")
print(trans_df.columns.tolist()[:10])  # 显示前10个特征
```
结果

```js
生成 2813 个特征:
['category=BLACK VOICES', 'category=BUSINESS', 'category=COMEDY', 'category=ENTERTAINMENT', 'category=FOOD & DRINK', 'category=HEALTHY LIVING', 'category=HOME & LIVING', 'category=PARENTING', 'category=PARENTS', 'category=POLITICS']
```

```js
from mlxtend.frequent_patterns import apriori

def run_apriori(trans_df, min_support=0.05):
    """
    严格按课程伪代码实现Apriori算法
    参数:
        min_support: 完全按课程定义的支持度阈值
    返回:
        - 频繁项集DataFrame (含support列)
    """
    # 直接调用mlxtend实现 (与课程伪代码逻辑一致)
    frequent_itemsets = apriori(
        trans_df, 
        min_support=min_support,
        use_colnames=True,  # 保留原始项集名称
        max_len=None        # 不限制项集长度
    )
    
    # 按课程要求添加length列
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    
    print(f"\n生成的频繁项集 (min_support={min_support}):")
    print(frequent_itemsets.sort_values('support', ascending=False).head())
    return frequent_itemsets

# 执行Apriori算法 (参数与课程示例一致)
frequent_itemsets = run_apriori(trans_df, min_support=0.03)
```

结果·

```js

生成的频繁项集 (min_support=0.03):
     support                                 itemsets  length
10  0.539238                     (sentiment=negative)       1
12  0.355381                     (sentiment=positive)       1
5   0.316143                      (category=POLITICS)       1
18  0.232063  (sentiment=negative, category=POLITICS)       2
1   0.227578                 (category=ENTERTAINMENT)       1
```



```js
from mlxtend.frequent_patterns import association_rules

def generate_rules(frequent_itemsets, min_confidence=0.7):
    """
    严格按课程定义的指标生成规则:
    - support: P(X ∪ Y)
    - confidence: P(Y|X) = support(X ∪ Y) / support(X)
    - lift: P(X ∪ Y) / (P(X)P(Y))
    """
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )
    
    # 按课程要求计算lift (已自动计算)
    rules = rules.sort_values(['lift', 'confidence'], ascending=False)
    
    print(f"\n生成的关联规则 (min_confidence={min_confidence}):")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
    return rules

# 生成规则 (参数与课程示例一致)
rules = generate_rules(frequent_itemsets, min_confidence=0.5)
```
结果

```js
生成的关联规则 (min_confidence=0.5):
                        antecedents                              consequents  \
1                      (entity=gop)                      (category=POLITICS)   
6  (sentiment=negative, word=trump)                      (category=POLITICS)   
3                      (word=trump)                      (category=POLITICS)   
8                      (word=trump)  (sentiment=negative, category=POLITICS)   
0           (category=FOOD & DRINK)                     (sentiment=positive)   

    support  confidence      lift  
1  0.034753    0.968750  3.064273  
6  0.036996    0.891892  2.821162  
3  0.052691    0.886792  2.805031  
8  0.036996    0.622642  2.683074  
0  0.033632    0.638298  1.796094
```


```js
def analyze_rules(rules):
    """按课程要求分析规则"""
    # 1. 筛选有效规则 (lift > 1表示正相关)
    meaningful_rules = rules[rules['lift'] > 1].copy()
    
    # 2. 按课程示例添加规则长度
    meaningful_rules['antecedent_len'] = meaningful_rules['antecedents'].apply(lambda x: len(x))
    meaningful_rules['consequent_len'] = meaningful_rules['consequents'].apply(lambda x: len(x))
    
    # 3. 按课程分类展示
    print("\n=== 正相关规则 (lift > 1) ===")
    print(meaningful_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
    
    # 4. 情感相关规则 (按课程项目要求)
    sentiment_rules = meaningful_rules[
        meaningful_rules['consequents'].apply(
            lambda x: any('sentiment=' in item for item in x)
        )
    ]
    
    print("\n=== 情感关联规则 ===")
    print(sentiment_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# 执行分析
analyze_rules(rules)
```



结果·

```js
=== 正相关规则 (lift > 1) ===
                        antecedents                              consequents  \
1                      (entity=gop)                      (category=POLITICS)   
6  (sentiment=negative, word=trump)                      (category=POLITICS)   
3                      (word=trump)                      (category=POLITICS)   
8                      (word=trump)  (sentiment=negative, category=POLITICS)   
0           (category=FOOD & DRINK)                     (sentiment=positive)   

    support  confidence      lift  
1  0.034753    0.968750  3.064273  
6  0.036996    0.891892  2.821162  
3  0.052691    0.886792  2.805031  
8  0.036996    0.622642  2.683074  
0  0.033632    0.638298  1.796094  

=== 情感关联规则 ===
                       antecedents                              consequents  \
8                     (word=trump)  (sentiment=negative, category=POLITICS)   
0          (category=FOOD & DRINK)                     (sentiment=positive)   
2              (category=POLITICS)                     (sentiment=negative)   
7  (category=POLITICS, word=trump)                     (sentiment=negative)   
5                     (word=trump)                     (sentiment=negative)   

    support  confidence      lift  
8  0.036996    0.622642  2.683074  
0  0.033632    0.638298  1.796094  
2  0.232063    0.734043  1.361260  
7  0.036996    0.702128  1.302075  
5  0.041480    0.698113  1.294630
```


```js
import matplotlib.pyplot as plt

def plot_rules(rules, top_n=10):
    """按课程项目示例绘制规则热力图"""
    if len(rules) == 0:
        print("无有效规则可可视化")
        return
    
    # 准备数据 (按课程示例格式)
    plot_data = rules.head(top_n).copy()
    plot_data['rule'] = plot_data.apply(
        lambda row: f"{set(row['antecedents'])} → {set(row['consequents'])}",
        axis=1
    )
    
    # 绘制热力图 (完全按课程风格)
    plt.figure(figsize=(10, 6))
    plt.scatter(
        plot_data['support'],
        plot_data['confidence'],
        s=plot_data['lift'] * 100,  # 点大小表示lift
        c=plot_data['lift'],        # 颜色表示lift
        alpha=0.6,
        cmap='viridis'
    )
    
    # 添加标签 (按课程示例)
    for i, row in plot_data.iterrows():
        plt.annotate(
            row['rule'],
            (row['support'], row['confidence']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    plt.colorbar(label='Lift')
    plt.xlabel('Support (P(X∪Y))', fontsize=12)
    plt.ylabel('Confidence (P(Y|X))', fontsize=12)
    plt.title('Association rule heat map (size/color indicates lift)', fontsize=14)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

# 可视化
plot_rules(rules[rules['lift'] > 1])
```

![image](https://github.com/user-attachments/assets/cf1618ef-e651-4f9e-8031-44a23397ae49)


```js
def mine_association_rules(trans_df, min_support=0.05, min_threshold=0.5):
    """
    执行关联规则挖掘
    参数:
        trans_df: one-hot编码的交易数据
        min_support: 最小支持度
        min_threshold: 最小提升度阈值
    """
    # 1. 找出频繁项集
    frequent_itemsets = apriori(trans_df, min_support=min_support, use_colnames=True)
    print(f"\nFound {len(frequent_itemsets)} frequent itemsets (min_support={min_support})")
    
    if len(frequent_itemsets) == 0:
        print("没有找到频繁项集，请降低min_support阈值")
        return None
    
    # 2. 生成关联规则
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
    print(f"Generated {len(rules)} association rules (min_threshold={min_threshold})")
    
    if len(rules) == 0:
        print("没有生成关联规则，请降低min_threshold阈值")
        return None
    
    # 3. 过滤并排序规则
    interesting_rules = rules[
        (rules['consequents'].apply(lambda x: any('sentiment=' in item for item in x))) &
        (rules['lift'] > 1)
    ].sort_values(['lift', 'confidence'], ascending=False)
    
    return interesting_rules

# 执行关联规则挖掘
rules = mine_association_rules(trans_df, min_support=0.03, min_threshold=0.7)

# 显示最有意义的规则
if rules is not None and len(rules) > 0:
    pd.set_option('display.max_colwidth', 100)
    print("\nTop 10 Interesting association rules:")
    display(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
else:
    print("\n没有发现有趣的关联规则")
```


![image](https://github.com/user-attachments/assets/954de244-95f3-4560-be82-0e623bb5286a)

 
```js
import networkx as nx
import matplotlib.pyplot as plt

def visualize_rules(rules, top_n=10):
    """可视化关联规则网络图"""
    if rules is None or len(rules) == 0:
        print("没有可可视化的规则")
        return
    
    # 准备数据
    top_rules = rules.head(top_n)
    graph = nx.DiGraph()
    
    # 添加节点和边
    for _, row in top_rules.iterrows():
        antecedents = ', '.join(list(row['antecedents']))
        consequents = ', '.join(list(row['consequents']))
        weight = row['lift']
        
        graph.add_edge(antecedents, consequents, weight=weight)
    
    # 绘制网络图
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, k=0.5)
    
    # 节点大小基于度数
    node_sizes = [3000 * graph.degree(node) for node in graph.nodes()]
    
    # 边宽度基于lift值
    edge_widths = [2 * row['weight'] for _, _, row in graph.edges(data=True)]
    
    # 绘制
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, alpha=0.7,
                          node_color='skyblue')
    nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.5,
                          edge_color='gray', arrowsize=20)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold')
    
    # 添加标题和说明
    plt.title(f"Top {top_n} Sentiment association rules (edge ​​width represents lift value)", fontsize=14)
    plt.axis('off')
    
    # 添加图例
    plt.text(0.5, -0.1, 
             "Arrow direction: Precondition → Result\nNode size: Number of connections\nEdge width: Rule lift value",
             ha='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()

# 可视化规则
visualize_rules(rules)

# 分析特定情感类别的规则
if rules is not None:
    for sentiment in ['positive', 'negative', 'neutral']:
        sent_rules = rules[rules['consequents'].apply(
            lambda x: f"sentiment={sentiment}" in x)]
        
        if len(sent_rules) > 0:
            print(f"\n与'{sentiment}'情感相关的规则:")
            display(sent_rules[['antecedents', 'consequents', 
                              'support', 'confidence', 'lift']].head(5))
```

![image](https://github.com/user-attachments/assets/d4a0f2ce-6c56-42e2-893a-b7dacf30b50d)

![image](https://github.com/user-attachments/assets/f7e7d16b-57a7-44f6-9188-8d7f8069abd3)

```js
# 1. 数据准备
print("步骤1: 准备关联规则挖掘数据...")
transactions, trans_df = prepare_association_data(df)

# 2. 挖掘关联规则
print("\n步骤2: 挖掘关联规则...")
rules = mine_association_rules(trans_df, 
                              min_support=0.03, 
                              min_threshold=0.7)

# 3. 可视化与分析
if rules is not None and len(rules) > 0:
    print("\n步骤3: 可视化与分析...")
    visualize_rules(rules)
    
    # 保存结果
    rules.to_csv("sentiment_association_rules.csv", index=False)
    print("\n关联规则已保存到 sentiment_association_rules.csv")
else:
    print("\n没有发现显著的关联规则")

```

结果

```js
步骤1: 准备关联规则挖掘数据...

步骤2: 挖掘关联规则...

Found 25 frequent itemsets (min_support=0.03)
Generated 24 association rules (min_threshold=0.7)

步骤3: 可视化与分析...
```
![image](https://github.com/user-attachments/assets/82de8446-6b9e-4c69-8a18-fc08a4a3ad3d)

关联规则已保存到 sentiment_association_rules.csv
# 基于情感，实体和内容的推荐系统

# 基于关联规则的挖掘系统

