
在COMP7630网络智能及其应用课程（2024 - 2025学年第二学期）中，老师对开发项目小组论文提出了多方面要求，从项目目标、内容涵盖到成果提交，均有明确规定。
1. **项目目标**：开发一个软件原型，将课程所学的网络智能相关技术应用于实际数据，通过解决实际问题、分析数据获取有价值的见解。
2. **内容涵盖**
    - **问题识别**：阐述获取特定在线数据的动机，以及分析这些数据期望获得的见解。例如，如果选择分析社交媒体数据，需说明为何关注该数据，是为了研究用户行为模式还是其他目的，以及期望从数据中发现什么。
    - **数据获取**：从多种在线来源获取数据，可通过网络爬虫或API等方式。如利用LinkedIn API获取人员资料、通过Github API获取开发项目细节，也可以从Spotify、新闻网站、Tripadvisor、Twitter、IMDB等平台获取数据，还能从古登堡项目收集无版权书籍或从Kaggle等网站下载数据集。
    - **数据预处理**：对获取的数据进行清理，包括去除噪声数据、处理缺失值、标准化数据格式等，以提高数据质量，为后续分析做准备。
    - **数据分析**：运用合适的方法对预处理后的数据进行分析，挖掘数据中的潜在信息和规律，获取有价值的见解。
    - **结果可视化**：通过图表或表格清晰展示分析结果，使数据信息更直观易懂。
3. **成果提交**
    - **小组项目报告**：按“WI_GroupX_ProjectReport.pdf”格式提交电子版报告，并提供项目代码和数据的ZIP文件链接（如OneDrive、Google Drive等云平台链接，ZIP文件名为“WI_GroupX_software.zip”）。报告需涵盖项目各步骤细节和结果讨论，结构包括小组项目标题、成员信息、摘要、动机、各步骤详细内容、结果讨论与解读、结论和参考文献。报告不超过10页（1.5倍行距，12磅字体，页边距2厘米，标题、小组信息、目录和参考文献不计入页数），可适当使用图表但不能过度。报告末尾需包含小组成员贡献详情表。
    - **个人报告**：每人按“IDXXXXXXXX_Report.pdf”格式提交电子版报告，内容为个人在项目中的见解、经验教训以及对项目新可能性的思考，1页（格式同小组项目报告要求）。
    - **视频展示**：每组提交一个MP4格式视频链接，视频时长不超过15分钟，使用幻灯片展示工作内容并讲解。要求所有小组成员参与讲解，确保讲解和谐流畅。视频需为标准MP4格式，可使用ActivePresenter或其他录屏软件录制，上传至云平台后提供可访问链接。
4. **其他要点**：小组所有成员都应参与问题识别和数据源确定，不同成员可负责项目的不同部分。提出越新颖、创新的应用，项目得分越高。 

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
