---
title: "Introduction to Bioinformatics / Data Science"
date: 04/02/2020
---


```{r 11, include=FALSE, cache=TRUE}
library(knitr)
knitr::opts_chunk$set(cache=TRUE, fig.pos="center")
```

# Introduction - What is Machine Learning?

At the intersection between Applied Mathematics, Computer Science, and Biological Sciences is a subset of knowledge known as Bioinformatics. Some find Bioinformatics and its cousin Data Science difficult to define. However, the most ubiquitous pictograph of Bioinformatics and Data Science may say a thousand words even if we cannot. See Figure 1. What separates these two pursuits is the domain knowledge that each focus on. Where Data Science may focus on understanding buying habits of individuals, Bioinformatics tends to focus on the understanding of DNA and its relevance to disease. It is now common to find new fields emerging such as the study of chemistry using applied mathematics and computers which beget the field of Chemo-informatics.[^15] [^16] Today, since hospitals are warehouses of modern medical records and knowledge they make career paths such as Healthcare-Informatics possible.[^17]

[^15]:https://www.acs.org/content/acs/en/careers/college-to-career/chemistry-careers/cheminformatics.html

[^16]:https://jcheminf.biomedcentral.com/

[^17]:https://www.usnews.com/education/best-graduate-schools/articles/2014/03/26/consider-pursuing-a-career-in-health-informatics
 
![Venn Diagrams of Bioinformatics And Data Science](./00-data/10-images/Venn-diagram-original-768x432.png)

[^11]

[^11]:http://omgenomics.com/what-is-bioinformatics/

Generally speaking, Bioinformatics derives its knowledge from various items. To start our data may consist of numbers, text and even pictures as input. Numerical values may be from scientific sensors or instrumentation, such as Next Generation[^127] DNA sequence data.[^12] Text data is sourced from articles/books or even databases of scientific articles like PubMed which show the inter-connectivity of thought and research.[^13] Graphics and pictures include graphical forms, such as relationship data that appears in genomic or proteomic or metabolic databases.[^14]

[^12]:http://www.genome.ucsc.edu/

[^13]:https://arxiv.org/
 
[^14]:https://www.ebi.ac.uk/training/online/course/proteomics-introduction-ebi-resources/what-proteomics

[^127]:What is Next-Generation DNA Sequencing?, https://www.ebi.ac.uk/training/online/course/ebi-next-generation-sequencing-practical-course/what-you-will-learn/what-next-generation-dna-

However, Bioinformatics and Data Science imply a process as well. One piece of this process is Reproducible Research.[^128] Reproducible Research and replication are linked but it goes beyond the ability of the work to be *independently verified*. The computer code and data must be provided. There locations explicitly spelled out such that other scientists may find and carry on the work and check all its calculations and methodology. Now that computers play such a large role procedurally the smallest error in a spreadsheet may plague the overall conclusion. Such is the case of the University of Mass. at Amherst found errors in a Harvard research paper.[^129]

[^129]:https://www.chronicle.com/article/UMass-Graduate-Student-Talks/138763

[^128]:Roger Peng, The Real Reason Reproducible Research is Important, https://simplystatistics.org/2014/06/06/the-real-reason-reproducible-research-is-important/

## Machine Learning Is?

>"Machine learning is essentially a form of applied statistics with increased emphasis on the use of computers to statistically estimate complicated functions and a decreased emphasis on proving confidence intervals around these functions"
>
>--- Ian Goodfellow, et al[^18]

[^18]:Ian Goodfellow, Yoshua Bengio, Aaron Courville, 'Deep Learning', MIT Press, 2016, http://www.deeplearningbook.org

### What is Predictive Modeling?

Although what this paper discusses is Predictive Modeling, it is under the umbrella of Machine Learning. The term 'Predictive Modeling' should bring to mind work in the computer science field, also called Machine Learning (ML), Artificial Intelligence (AI), Data Mining, Knowledge discovery in databases (KDD), and possibly even encompassing Big Data as well.

>"Indeed, these associations are appropriate, and the methods implied by these terms are an integral piece of the predictive modeling process. But predictive modeling encompasses much more than the tools and techniques for uncovering patterns within data. The practice of predictive modeling defines the process of developing a model in a way that we can understand and quantify the model's prediction accuracy on future, yet-to-be-seen data."
>
>Max Kuhn[^19]

[^19]:Max Kuhn, Kjell Johnson, Applied Predictive Modeling, Springer, ISBN:978-1-4614-6848-6, 2013

As an aside, I use `Predictive Modeling` and `Machine Learning` interchangeably in this document.

In the booklet entitled "The Elements of Data Analytic Style," [^110] there is an useful checklist for the uninitiated into the realm of science report writing and, indeed, scientific thinking. A shorter, more succinct listing of the steps, which I prefer, and is described by Roger Peng in his book, The Art Of Data Science. The book lists what he describes as the "Epicycle of Analysis." [^111]

[^110]:Jeff Leek, The Elements of Data Analytic Style, A guide for people who want to analyze data., Leanpub Books, http://leanpub.com/datastyle, 2015

[^111]:Roger D. Peng and Elizabeth Matsui, The Art of Data Science, A Guide for Anyone Who Works with Data, Leanpub Books, http://leanpub.com/artofdatascience, 2015

**The Epicycle of Analysis**

1. Stating and refining the question,
1. Exploring the data,
1. Building formal statistical models,
1. Interpreting the results,
1. Communicating the results.

**Predictive Modeling**

In general, there are three types of Predictive Modeling or Machine Learning approaches;

1. Supervised,
2. Unsupervised,
3. Reinforcement.

Due to the fact this paper only uses Supervised & Unsupervised learning and for the sake of this brevity, I discuss only the first two types of Predictive Models.

### Supervised Learning

In supervised learning, data consists of observations $x_i$ (where $X$ may be a matrix of $\Re$ values) AND a corresponding label, $y_i$. The label $y$ maybe anyone of $C$ classes. In our case of a binary classifier, we have {'Is myoglobin', 'Is control'}.

**Data set**:

- $(X_1, y_1), (X_2 , y_2), ~. . ., ~(X_N , y_N);$

- $y \in \{1, ..., ~C\}$, where $C$ is the number of classes

A machine learning algorithm determines a pattern from the input information and groups this with its necessary title or classification. 

One example might be that we require a machine that separates red widgets from blue widgets. One predictive algorithm is called a K-Nearest Neighbor (K-NN) algorithm. K-NN looks at an unknown object and then proceeds to calculate the distance (most commonly, the euclidean distance) to the $K$ nearest neighbors. If we consider the figure below and choose $K$ = 3, we would find a circumstance as shown. In the dark solid black on the K-Nearest-Neighbor figure, we find that the green widget is nearest to two red widgets and one blue widget. In the voting process, the K-NN algorithm (2 reds vs. 1 blue) means that the consignment of our unknown green object is red. 

For the K-NN algorithm to function, the data optimally most be complete with a set of features and a label of each item. Without the corresponding label, a data scientist would need different criteria to track the widgets.

Five of the six algorithms that this report investigates are supervised. Logit, support vector machines, and the neural network that I have chosen require labels for the classification process.

![Example of K-Nearest-Neighbor](./00-data/10-images/K-Nearest-Neighbor.50.png)

[^112]

[^112]:https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

#### What is a shallow learner? {-}

Let us investigate the K-NN algorithm and figure 2.2 (K-Nearest-Neighbor) a little further. If we change our value of $K$ to 5, then we see a different result. By using $K = 5$, we consider the out dashed-black line. This more considerable $K$ value contains three blue widgets and two red widgets. If we ask to vote on our choice, we find that 3 blue beats the 2 red, and we assign the unknown a BLUE widget. This assignment is the opposite of the inner circle. 

If a researcher were to use K-NN, then the algorithm would have to test many possible $K$ values and compare the results, then choose the $K$ with the highest accuracy. However, this is where K-NN falters. The K-NN algorithm needs to keep all data points used for its initial training (accuracy testing). Any new unknowns could be conceivably tested against any or all the previous data points. The K-NN does use a generalized rule that would make future assignments quick on the contrary. It must memorize all the points for the algorithm to work. K-NN cannot delete the points until it is complete. It is true that the algorithm is simple but not efficient. Matter and fact, as the number of feature dimensions increases, this causes the complexity (also known as Big O) to rise. The complexity of K-NN is $O$(K-NN) $\propto nkd$.

Where $n$ is the number of observations, $k$ is the number of nearest neighbors it must check, and d is the number of dimensions.[^113]

[^113]:Olga Veksler, Machine Learning in Computer Vision, http://www.csd.uwo.ca/courses/CS9840a/Lecture2_knn.pdf

Given that K-NN tends to 'memorize' its data to complete its task, it is considered a lazy and shallow learner. Lazy indicates that the decision is left to the moment a new point is learned of predicted. If we were to use a more generalized rule, such as {Blue for ($x \leq 5$)} this would be a more dynamic and more in-depth approach by comparison.

### Unsupervised Learning

In contrast to the supervised learning system, unsupervised learning does not require or use a **label** or **dependent variable**. 

**Data set**:

- $(X_1), (X_2), ~. . ., ~(X_N)$

- **No Y**

where $X$ may represent a matrix of $m$ observations by $n$ features with $\Re$ values.

Principal Component Analysis is an example of unsupervised learning, which we discuss in more detail in chapter 3. The data, despite or without its labels, are transformed to provide maximization of the variances in the dataset. Yet another objective of Unsupervised learning is to discover "interesting structures" in the data.[^114] There are several methods that show structure. These include clustering, knowledge discovery of latent variables, or discovering graph structure. In many instances and as a subheading to the aforementioned points, unsupervised learning can be used for dimension reduction or feature selection.

[^114]:Kevin Murphy, Machine learning a probabilistic perspective, 2012, ISBN 978-0-262-01802-9

Among the simplest unsupervised learning algorithms is K-means. K-means does not rely on the class labels of the dataset at all. K-means may be used to determine any number of classes despite any predetermined values. K-means can discover clusters later used in classification or hierarchical feature representation. K-means has several alternative methods but, in general, calculates the distance (or conversely the similarity) of observations to a mean value of the $K$th grouping. The mean value is called the center of mass, the Physics term that provides an excellent analogy since the center of mass is a weighted average. By choosing a different number of groupings (values of $K$, much like the K-NN), then comparing the grouping by a measure of accuracy, one example being, mean square error. 

![Example of K-Means](./00-data/10-images/k-means-2.50.png)

[^115]

[^115]:https://www.slideshare.net/teofili/machine-learning-with-apache-hama/20-KMeans_clustering_20

### Five Challenges In Predictive Modeling

To many predictive modeling is a panacea for all sorts of issues. Although it does show promise, some hurdles need research. Martin Jaggi[^116] has summarized four points that elucidate current problems in the field that need research. To this list, I have added one more point, which is commonly called the *Vaiance-Bias Tradeoff*.

**Problem 1:** The vast majority of information in the world is unlabeled, so it would be advantageous to have a good Unsupervised machine learning algorithms to use,

**Problem 2:** Algorithms are very specialized, too specific,

**Problem 3:** Transfer learning to new environments,

**Problem 4:** Scale, the scale of information is vast in reality, and we have computers that work in gigabytes, not the Exabytes that humans may have available to them. The scale of distributed Big Data,

[^116]:https://www.machinelearning.ai/machine-learning/4-big-challenges-in-machine-learning-ft-martin-jaggi-2/

The specific predictive models which are executed in this report are discussed in further detail in their own sections. 

**Problem 5:** Bias-Variance Trade-Off.

The ability to generalize is a key idea in predictive modeling. This idea harkens back to freshman classes where one studied Student's t-test and analysis of variance.

\begin{equation}
E \left[ \left( y_0 - \hat f(x_0) \right )^2 \right ] = Var ( \hat f(x_0)) + \left [Bias (\hat f(x_0)) \right]^2 + Var(\epsilon)
\end{equation}

---

![Bias-Variance Tradeoff](./00-data/10-images/bias-variance-tradeoff.50.png)

The bias-variance dilemma can be stated as follows.[^67]

[^67]:Trevor Hastie, Robert Tibshirani, Jerome Friedman, The Elements of Statistical Learning; Data Mining, Inference, and Prediction, https://web.stanford.edu/~hastie/ElemStatLearn/, 2017

>1. Models with too few parameters are inaccurate because of a large bias: they lack flexibility.
>
>2. Models with too many parameters are inaccurate because of a large variance: they are too sensitive to the sample details (changes in the details will produce huge variations).
>
>3. Identifying the best model requires controlling the “model complexity”, i.e., the proper architecture and number of parameters, to reach an appropriate compromise between bias and variance.

One very good example is seen in figure 2.4, *Bias-Variance Tradeoff*. By considering the yellow-orange line we find a simple slope intercept model ($y \propto k \cdot x$) where the variance is high but the bias low and is not flexible enough. This is called underfitting. Looking at the green-blue line we see it follows the data set much more closely, e.g. ($y \propto k \cdot x^8$). Here the variance is very low but common sense tells us that the line is overfit and would not generalize well in a real world setting. Finally leaving us with the black line which does not appear to have too many parameters, ($y \propto k \cdot x^3$).

## Research Description

Is there a correlation between the data points, which are outliers from principal component analysis (PCA), and six types of predictive modeling? 

This experiment is interested in determining if PCA would provide information on the false-positives and false-negatives that were an inevitable part of model building and optimization. The six predictive models that have chosen for this work are Logistic Regression, Support Vector Machines (SVM) linear, polynomial kernel, and radial basis function kernel, and a Neural Network.

I have studied six different M.L. algorithms using protein amino acid percent composition data from two classes. Class number 1 is my positive control which is a set of Myoglobin proteins, while the second class is a control group of human proteins that do not have Fe binding centers.

| Group     |    Class | Number of Class | Range of Groups |
| :-------- | -------: | --------------: | --------------: |
| Controls  | 0 or (-) |            1216 |    1, ..., 1216 |
| Myoglobin | 1 or (+) |            1124 | 1217, ..., 2340 |

It is common for Data Scientists to test their data sets for feature importance and feature selection. One test that has interested this researcher is Principal component analysis. It can be a useful tool. PCA is an unsupervised machine learning technique which "reduces data by geometrically projecting them onto lower dimensions called principal components (PCs), with the goal of finding the best summary of the data using a limited number of PCs." [^117] However, the results that it provides may not be immediately intuitive to the layperson.

[^117]:Jake Lever, Martin Krzywinski, Naomi Altman, Principal component analysis, Nature Methods, Vol.14 No.7, July 2017, 641-2

How do the advantages and disadvantages of using PCA compare with other machine learning techniques? The advantages are numerable. They include dimensionality reduction and filtering out noise inherent in the data, and it may preserve the global structure of the data. Does the global and graphical structure of the data produced by the first two principal components provide any insights into how the predictive models of Logistic Regression, Neural Networks utilizing auto-encoders, Support Vector Machines, and Random Forest? In essence, is PCA sufficiently similar to any of the applied mathematics tools of more advanced approaches? Also, this work is to teach me machine learning or predictive modeling techniques.

The data for this study is from the Uniprot database. From the Uniprot database was queried for two protein groups. The first group was Myoglobin, and the second was a control group comprised of human proteins not related to Hemoglobin or Myoglobin. See Figure 1.5, *Percent Amino Acid Composition*. There have been a group of papers that are striving to classify types of proteins by their amino acid structure alone. The most straightforward classification procedures involve using the percent amino acid composition (AAC). The AAC is calculated by using the count of an amino acid over the total number in that protein.

- Percent Amino Acid Composition: 

\begin{equation} 
\%AAC_X ~=~ \frac{N_{Amino~Acid~X}}{Total ~ N ~ of ~ AA}
\end{equation}

The Exploratory Data Analysis determines if features were skewed and needed must be transformed. In a random system where amino acids were chosen at random, one would expect the percent amino acid composition to be close to 5%. However, this is far from the case for the Myoglobin proteins or the control protein samples. On top of this the differences between the myoglobin and control proteins can be as high as approximately 5% with the amino acid Lysine, K.

![Mean Percent Amino Acid Compositions For Control And Myoglobin](./00-data/10-images/c_m_Mean_AAC.png)

### Exploratory Data Analysis (EDA)

During EDA, the data is checked for irregularities, such as missing data, outliers among features, skewness, and visually for normality using QQ-plots. The only irregularity that posed a significant issue was the skewness of the amino acid features. Many of 20 amino acid features had a significant number of outliers, as seen by Boxplot analysis. However, only three features had skew, which might have presented a problem. Dealing with the skew of the AA was necessary since Principal Component Analysis was a significant aspect of this experiment.

Testing determined earlier that three amino acids (C, F, I) from the single amino acid percent composition needs transformation by using the square root function. The choice of transformations was natural log, log base 10, squaring ($x^2$), and using the reciprocal ($1 / x$) of the values. The square root transformation lowered the skewness to values of less than 1.0 from high points of greater than 2 in all three cases to {-0.102739 $\leq$ skew after transformation $\leq$ 0.3478132}.

| Amino Acid       | Initial skewness | Skew after square root transform |
| :--------------- | :--------------: | :------------------------------: |
| C, Cysteine      |     2.538162     |           0.347813248            |
| F, Phenolalanine |     2.128118     |           -0.102739748           |
| I, Isoleucine    |     2.192145     |           0.293474879            |

Three transformations take place for this dataset. 

`~/00-data/02-aac_dpc_values/c_m_TRANSFORMED.csv` and used throughout the rest of the analysis.

All work uses R[^118], RStudio[^119] and a machine learning library/framework `caret`.[^120]

[^118]:https://cran.r-project.org/

[^119]:https://rstudio.com/

[^120]:http://topepo.github.io/caret/index.html

