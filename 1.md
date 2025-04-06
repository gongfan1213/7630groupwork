### **Group Project Report Outline (中英对照大纲)**  
**（严格遵循课程要求，基于开发过程文档优化）**  

---

#### **Title | 标题**  
- **English**:  
  *"Advanced News Analytics System: Integrating Sentiment Mining, Topic Modeling, and Real-time Visualization on HuffPost Dataset"*  
- **中文**:  
  *"高级新闻分析系统：基于HuffPost数据的情感挖掘、主题建模与实时可视化"*  

---

#### **1. Abstract | 摘要**  
- **English**:  
  Summarize the project’s objectives, methods (sentiment analysis, LDA, association rules), key results (e.g., POLITICS is most negative), and innovations (dynamic domain adaptation, irony detection).  
- **中文**:  
  概述项目目标（情感分析、主题建模、关联规则）、核心结果（如政治类新闻最负面）和创新点（动态领域适应、讽刺检测）。  

---

#### **2. Motivation & Problem Identification | 动机与问题定义**  
- **English**:  
  - **Why News Analytics?** Public opinion impact, manual analysis inefficiency.  
  - **Challenges**: Sarcasm detection, domain-specific sentiment bias, topic evolution.  
- **中文**:  
  - **分析动机**：舆情影响、人工分析效率低。  
  - **技术挑战**：讽刺识别、领域情感偏差、主题动态演化。  

---

#### **3. Data Acquisition & Preprocessing | 数据获取与预处理**  
- **3.1 Data Sources | 数据来源**  
  - Kaggle (2012–2022) + Web scraping (2022–2025).  
- **3.2 Preprocessing Steps | 预处理步骤**  
  - Tokenization, stopword removal, lemmatization, class balancing (oversampling).  

---

#### **4. Methodology & Implementation | 方法与实现**  
- **4.1 Topic Modeling (LDA) | 主题建模**  
  - Batch processing, 15 topics, pyLDAvis visualization.  
- **4.2 Ensemble Sentiment Analysis | 集成情感分析**  
  - VADER + TextBlob + BERT, entity-level sentiment mapping.  
- **4.3 Dynamic Domain Adaptation | 动态领域适应**  
  - Politics/TECH rules (e.g., "scandal" → penalty).  
- **4.4 Association Rule Mining | 关联规则挖掘**  
  - Recommendations (e.g., `category=ENVIRONMENT → sentiment=negative`).  

---

#### **5. Results & Visualization | 结果与可视化**  
- **5.1 Sentiment Distribution | 情感分布**  
  - 203K positive vs. 221K negative news.  
- **5.2 Topic-Term Heatmaps | 主题-词热力图**  
  - Top 10 words per topic (e.g., Topic 1: "trump, election, vote").  
- **5.3 Interactive Dashboards | 交互式看板**  
  - Plotly charts for entity sentiment trends.  

---

#### **6. Discussion & Innovation | 讨论与创新**  
- **6.1 Key Findings | 核心发现**  
  - TRAVEL most positive (0.16 avg), POLITICS most negative (-0.16 avg).  
- **6.2 Technical Innovations | 技术创新**  
  - RoBERTa irony detection, real-time alert system (threshold=-0.2).  

---

#### **7. Conclusion & Future Work | 结论与未来方向**  
- **English**:  
  - **Conclusion**: System enables efficient media monitoring.  
  - **Future**: Deploy as Flask API, integrate real-time Twitter data.  
- **中文**:  
  - **结论**：系统提升舆情分析效率。  
  - **未来方向**：部署为Flask服务，接入Twitter实时数据。  

---

#### **8. References | 参考文献**  
- Devlin et al. (2019) for BERT, Scikit-learn documentation.  

---

#### **9. Contribution Table | 贡献说明表**  
- **Format**:  
  | Member Name | Contributions |  
  |-------------|---------------|  
  | Alice       | Data scraping, LDA modeling |  
  | Bob         | Sentiment analysis, visualization |  

---

### **评分关键点覆盖**  
1. **完整性**：覆盖数据获取→分析→可视化全流程。  
2. **创新性**：动态领域适应、讽刺检测等超出基础要求。  
3. **可视化**：交互图表（Plotly）、热力图、词云。  
4. **格式规范**：10页内，含贡献表，标题/摘要不计页数。  

此大纲确保符合课程评分标准（60%报告分），突出技术深度与创新性。
