# Notebooks for experiments

## Background

In this project, we intend to propose a process of estimating physical stores' **marketing size** in certain areas based on certain data collected through public channels. This process could provide, especially for new comers in some offline large-consuming business tracks, an easy-to-get but intuitively and interprtable estimation on how promising the business will be.

## Method

### Define DNI

First of all, we can derive a store's **Daily net income (DNI)** from **Daily customer flow (DCF)** and **Per capita consumption (PCC)** by the following formula:

$$ DNI = DCF \times PCC $$

### Decompose DCF

We further decompose **DCF** by a furmula based on a **Region's customer flow (RCF)** and **Customer acquisition rate (CAR)**, where **Region's customer flow** is devided into 2 parts, local customer and visitors.

$$ RCF \approx Region\'s\ residents\ (num_{R}) + Region\'s Visitors\ (num_{V}) $$

with **$CAR_{i}$** for each group are denoted as $CAR_{R}$ and $CAR_{V}$ respectively, thus the total formula of **RCF** becomes:

$$ RCF = num_{R} \times CAR_{R} + num_{V} \times CAR_{V}$$

### Estimate CAR

We try to estimate **CAR** based on consuming logic of a customer. In general, we can translate **CAR** into **Probability of a customer choosing the store**, which may varies among different groups of people. However, due to difficulties in collecting precise data for each customer group, we simply treat the population of customers as a whole, and roughly estimate **CAR** by predicting a probability $P$. In this case, above formula of **RCF** becomes:

$$ RCF = (num_{R} + num_{V}) \times P$$

To predict $P$, we need to construct a model to include information about the features of a store. For example, **ratings** of other customers (suppose we have some online rating platform like [Dianping](https://www.dianping.com/) or [Zomato](https://www.zomato.com/), etc), store **location**, **products** or **services** the store offers and PCC as well could be in the customer's consideration. In this case, we can use some machine learning algorithms to model this decision process, e.g. Logistics Regression, Decision Tree, Random Forest, etc. Therefore, we can derive a pseudo model of estimating $P$:

$$ P = f(ratings, location, product features, PCC...)$$

PS: this P is a general estimation of the population, because our target is to predict the approximate preference of the customers. However, if we want more specific estimation, we may refer to some methods in *Recommendation System*, which can estimate more personalised preference for different customers, while may caused more cost in collecting required data as well.

### Modify ratings

Even though rating platforms like [Dianping](https://www.dianping.com/) or [Zomato](https://www.zomato.com/) provide rich information about the preference of other historical customers, sometimes some of this ratings can not be completely accepted. Especially for those stores with relative fewer votes, which suggests that those ratings could be easily influenced by certain individual's behaviors. Basically, we tend to believe those ratings with more votes, as a phenomenon that can be explained by "Law of large numbers" in statistics.

Therefore, this inspires us that, when using ratings to predict customers' preference, we need a rating modification process which will help regularized those ratings with fewer votes, in order for treating each store fairly.

In this case, we propose a modification method by traning a rating predictor model on samples with higher votes, then predict rating for those stores who lack as sufficient votes as the former "credible" samples, then perform a weighted sum for them based on the ground-truth rating and the predicted rating. This process can be expressed in following pseudo model:

$$ r_{predict} = f(store\ features)$$

$$ r\prime = w_{1} \cdot r_{ground\ truth} + w_{2} \cdot r_{predict} {\kern 20pt} s.t. w_{1}+w_{2}=1$$

Additionally, in order to test whether our results are serious, we may conduct some statistical analysis on the modified ratings. E.g. test whether stores in same area providing similar products and offering similar prices have the same rating, test whether rating among stores offer different products differs, etc.

## Summation

To sum up, to implement this market sizing design, we conduct the experiments in a sequence reversed to the above process, i.e. decompose the estimation into 4 parts: rating modification, user preference prediction, customer flow estimation and finally derive a predicting value of daily net income for all stores. Our experiment data is the [Zomato Bangalore Restaurants](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants/data).
