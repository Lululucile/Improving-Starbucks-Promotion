Part 1 Introduction
===
Part 1.1 Business Problem
---
Starbucks Corporation is an American multinational chain of coffeehouses and roastery reserves (About Us: Starbucks Coffee Company, 2022). As part of their pilot marketing campaign, different promotional offers were sent to a sample of Starbucks mobile APP members in a span of 30 days in the USA (Muliar, 2020). Inherently, individuals have distinct value consideration sets and respond differently to incentives (Abaluck and Adams-Prassl, 2021). Thus, to maximize return on marketing investments, it is crucial for Starbucks to only direct specific promotional campaigns to specific segments of customers who are most “reactive” to the promotion, to subsequently induce sales and profit advancements (Wong et al., 2005). Else, excessive marketing capital may be wasted on inattentive clients (those who do not value the particular promotion campaign), or profit margins may be eroded by giving a too generous reward (a lower discount may suffice for certain groups for inducing sales) (Wong et al., 2005). For future targeted marketing campaigns, this pilot study data can be leveraged in building machine learning algorithms to accurately predict if a member is likely to complete the offer. In essence, the business problem this research focuses on can be stated as - Maximizing revenue of marketing campaign by direct targeting promotion-sensitive customers

Part 1.2 Data Mining Techniques
---
To understand Starbucks customers, mining techniques like k-means clustering and principal component analysis is performed. Additionally, to identify the promotion sensitive customers,classification and regression techniques are used to predict the probability or likelihood of customers completing the offer, keeping dependent variable as binary (0 or 1) i.e.. if the offer will be completed or not. Specifically, logistic regression, lasso, post lasso, classification trees, random forest are used in the study.

Part 2 Data Preparation
===
The Starbucks Customer Data dataset, available on Kaggle, consisted of 3 files:1) “portfolio.csv”Details of the ten unique offer ids (BOGO, informational, and discount) used by Starbucks in the pilot study. 2) “profile.csv” - Demographic information of the 17,000 members of the study. 3) “transcript.csv” - Details of each promotional offer that a member received with 306533 rows. (Exhibit 20) contains detailed elaborations of each column key.

__Data cleaning/ identification__: Identified and removed 397 duplicates in the “Transcript.csv” based on “person,” “event,” “value,” and “time”. Data types of variables were transformed like "became member on" was converted to date type and saved the column as 'bmo'; 2175 member IDs (12% of the data) was identified as bad quality data with following all three conditionsmissing income, age equal to 118, and missing gender value. Even though 12% is a high number, to maintain data integrity and to not compromise on the data quality, 2175 person_id data was removed for analysis purposes.

__Data Transformation__: Each offer time stamp (by customer) is provided as a row in transcript data. There are 3 levels to each offer- offer received, offer viewed, and offer completed. Note that there are transactional rows for each customer, independent of promotional data. Breaking down the single string of information previously kept in the "value" column (from the transcript table) into three columns: Those that received or viewed the offer (134002 records); Those whocompleted the offer and received corresponding reward (33182 records); Direct purchases, independent from promotional offers, and their transaction 'amount' (138953 records). Furthermore, records related to a specific offer type in the “Portfolio.csv,” and customer demographics like age, gender, income, and member since, from “Profile.csv” are merged into the transcript table to get one dataset that provides information across both- customers and offers. Offer names were also redefined from “A” to “J” instead of the original alphanumeric type. Moreover, data was in a longer format i.e., timestamp for each offer was recorded in rows. Data was changed from a longer format to a wider format such that there will be only one occasion for each offer and customer level. Offers received, viewed, and completed were moved from rows to columns. This will help to use these data points as input variables at modeling stage. Dummy variables and new variables were computed to get more metrics. Example: number of years of membership was calculated from membership date to assess member’s loyalty, nonpromotional spend, frequency of interactions with Starbucks, and conversion rates.

Part 3: Data Understanding
===
3.1 EDA on customer profile data
---
Looking at the overall demographics of the data, we can assess the customer demographics characteristics of Starbucks members. The sample appears to be slightly more male-dominant than female (Exhibit1), with a mean age of 54 and a standard deviation of 17.5 (Exhibit2). Midage adults become largest target customers (Exhibit3). The customers' income ranges from $30,000 to $120,000, with a mean of 64,349. (Exhibit4 & 5). Looking into members by year, Starbucks continuously expanded its customer base from 2013 to 2017, with the greatest numberof members enrolled in 2017, followed by 2016, and a decrease in new members enrolled in 2018. (Exhibit 6).

3.2 EDA on Offer portfolio and transcript data
---
Most Starbucks sales within the 30 days period were from non-promotional transactions, while about 20.5% of promotional offers directly induced Starbucks sales. The Completion Rate is defined as the number of completed offers divided by the number of offers received. Those who joined in 2016 have the highest completion rate (Exhibit 9). (Exhibit 10) reveals offer G performs the best, which is a discount offer sent out through the web. Email, mobile, social, and informational offers have no completion rates suggesting that these are not lucrative. (Exhibit 11) shows that using all four channels will give the highest completion rate. Across the 30-day pilot run, it was found that there are six peaks of events (Exhibit 13) suggesting higher activity on certain days. (Exhibit 14) A high correlation between completed target and duration was seen, suggesting that offers with high duration have a longer influence on members.

![Pictures in Readme/3.2.png](https://github.com/Lululucile/Improving-Starbucks-Promotion/blob/main/Pictures%20in%20Readme/3.2.png)

3.3 Customer segmentation based on K-means
---
### 3.3.1 Data mining
For clustering, additional metrics as number of offers viewed and number of transactions was added to see clusters across members (Note: one outlier with spend as 1608.69 was removed). Six clusters were identified that explained 67% of the data (Exhibit 15-1).

### 3.3.2 Interpretation based on customer behaviors and suggestion
Based on the results (Exhibit 15-2): Cluster 1 (Promotion seekers- 23%) with high completion rate and average amount spend; Cluster 2 (Non-target group- 12%) with lowest completion rate and low spend; Cluster 3 (High Value customers- 17%) with highest spend and average completion rate; Cluster 4 (Offer viewers- 26%) with highest view rate but low completion rate; Cluster 5 (Frugal - 2%) with high number of transactions but of very small amounts; Cluster 6 (Regular customers- 20%) with average completion rate and medium/low spend; Based on cluster level view, Starbucks should focus on “Offer viewers” who engage with the campaign but don’t see value in the promotions. Additionally, using the devised models, Starbucks would be able to increase completion rates for “Promotion seekers” and “high value customers”. More details on demographics and standardized comparison by cluster in (Exhibit 15-2).

![Pictures in Readme/3.3.png](https://github.com/Lululucile/Improving-Starbucks-Promotion/blob/main/Pictures%20in%20Readme/3.3.png)

### 3.4 Principal Component Analysis
PCA suggested 3 factors explaining the variation in data. The first factor details more discountoriented deals with high duration, the second factor explains value deals that earn high rewards, and the third factor explains loyalty ie. high interactions, and spending at Starbucks. Graphs for the analysis on Exhibit 21.

Part 4- Modeling
===
__Mathematical Model__- Conversion or completion rate, widely used across the industry, can be used to understand Starbucks's marketing campaign success. However, organizations cannot target the entire customer base since sending promotions comes with a cost. To optimize marketing, Starbucks needs to identify the probability that customers would complete the offer and send offers to only those customers with higher likelihood to avail the offer- such that Starbucks reduces cost and maximizes returns by analyzing customers’ behavior and demographics. Following is a mathematical model to depict the identified business problem:

__max(E(Revenue))* = Prob(Completion of offer=1 | offer X, customer Y) + E( Value generated from offer | offer X, customer Y)__

_*Ignoring the value when completion of offer=0, as there is no additional revenue in such cases_

The core task was predicting the probability of a customer completing the offer, given the offer X. The target could have either completed the offer (1) or didn’t complete the offer (0). Regression and classification techniques were used to find the model that provides the best accuracy. Therefore, when Starbucks decides its target customers, it can balance between offer cost and completion rate to achieve the maximum revenue.

__Input variables__- Removed the offer_id as independent variable as Starbucks might choose to make some additions to current offer elements like increasing the duration in future marketing campaigns. Hence, offer details and customer demographics were used as independent variables for modeling. Demographic variables are- age, income, total spend during the marketing campaign, number of years since being a member, and number of interactions (completed offers+ non-promotional transaction); and offer variables are- rewards, min required to spend, channel, duration, number of offers received, and number of offers viewed.

__Modeling__- Modeling was performed on 80% of data- training data, and 20% of data was set aside to see the accuracy outside of the training set. 

__Five modeling techniques__ were used, along with a baseline model (null model):

1) logistic regression
2) logistic regression with interaction using lasso
3) logistic regression with interaction using post lasso
4) classification tree, random forest
5) average of models Lasso and post lasso

While random forest iterates over multiple trees to reduce overfitting and provides good accuracy across classification techniques. The alternatives could be KNN (k- nearest neighbors) that could potentially cause problems with scaling and less accuracy. More complex models are also used as alternatives- support vector machines (SVM), neural networks, random forest with higher number of trees that might provide higher accuracy, but these would take longer computational time, and result in more complexity.

Part 5- Evaluation
===
The sample was balanced at 50% of completes and seemed unbiased. __The baseline or null model shows that all customers will complete the offer with 50.06% accuracy__. Within the training set, 10-cross fold validation was performed across six models to remove bias and understand thebest-suited model. 

__Three metrics were captured for comparison:__

1) R square value
2) Average closeness of prediction from the true value (performance) 
3) Total accuracy [(True positive + True negative)/sample size] at 0.5 threshold

Logistic regression with interaction using lasso, Logistic regression with interaction using post lasso, and random forest showed very similar accuracy across the 10 folds. In an ideal scenario, models with higher accuracy on all three metrics would be chosen for test data. We chose the Random Forest model as the best-suited model for the problem at hand and achieved 88% accuracy on test data at 0.5 threshold with 66% R square, and 83% of closeness to the actual value.

![Pictures in Readme/5.png](https://github.com/Lululucile/Improving-Starbucks-Promotion/blob/main/Pictures%20in%20Readme/5.png)

Above graphs show the ROC curve and box plots of 10 cross validations across the three metrics. Exhibit 21 has bar graphs that includes the null model (baseline) for comparison.

Part 6 - Deployment
===
__How will the model be used?__ The developed model can be used by Starbucks to selectively target: 1) Customers who are more likely to respond to particular promotion types analyzed in this study,
0 787 4627whilst 2) Cut costs by not sending promotions to all customers (avoid sending redundant/ ineffective promotions). Effectively directing specific campaigns will be key to inducing larger, profit-maximizing sales volumes for Starbucks. Furthermore, marketing capital alleviated from this process can either be reinvested in developing new campaigns targeting segment or devoted/reinvested in providing incentives to reinforce other marketing objectives, such as customer retention.

Moreover, Cumulative response and lift curves can be used for effective targetingCR curve states that by targeting 40% of members, Starbucks can target 80% of potential customers, and lift curve states a similar approach by showing that if Starbucks targets 40% of customers, then lift of 2 can be achieved ie. can achieve twice as more potential customers than average.

![Pictures in Readme/6.1.png](https://github.com/Lululucile/Improving-Starbucks-Promotion/blob/main/Pictures%20in%20Readme/6.1.png)

__Issues and Assumptions__- However, before deploying the model, Starbucks should be aware that there was no information provided if customers would buy from Starbucks irrespective of whether a promotion was provided or not. There will be an underlying bias for those customers who are regular customers and would have gone to Starbucks even without the promotion. Previous transaction history is needed to understand how much “additional” revenue was gained from the marketing campaign. Additionally, the costs of each promotion are required to understand the profitability of each offer to provide remarks on the profit side. Since this information was missing, the current model focuses on maximizing revenue.

__Ethical considerations__ - There might be implications in using customer data for targeting promotion sensitive groups. In current data privacy norms, Starbucks may require further customer consent to use their transaction and demographic details for offers and targeted marketing to abide by data laws and follow ethical standards.

__Risks associated__ - Under the business context, models that help mitigate missed opportunities and lower marketing costs are the best while correctly predicting the probability of completion. From the above graph, as the threshold is changed, there is a tradeoff between True Positive Rate and False Positive Rate. If it is discovered that cost of targeting is greater than the missed opportunity (i.e., not targeting a potential customer), then Starbucks should increase the threshold to greater than 0.5, so to decrease the False positive. Similarly, if costs of targeting are lower than the missed opportunity cost, then Starbucks can reduce the threshold from 0.5 to decrease False Negatives (i.e., decrease missed opportunities). Since there is not enough information on the costs, adjusting the threshold (currently at 0.5) to account for the tradeoff between false negatives and false positives cannot be addressed at this moment.

![Pictures in Readme/6.2.png](https://github.com/Lululucile/Improving-Starbucks-Promotion/blob/main/Pictures%20in%20Readme/6.2.png)
