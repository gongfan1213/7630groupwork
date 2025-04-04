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
























 

