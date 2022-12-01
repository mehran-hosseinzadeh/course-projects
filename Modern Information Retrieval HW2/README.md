# Modern Information Retrieval HW2

In this project, using a subset og AG News dataset, I have performed various classification and clustering methods. The data contains news documents, each one consisting of a title, body, and a category (1: World, 2: Sports, 3: Business, 4: Science/Technology) in files ***train.json*** and ***validation.json***.

Documents are all transformed into tf-id representation.

The following classification methods are applied to these data:
- kNN (implemented from scratch)
-  Naive Bayes (implemented from scratch)
-  SVM (using built-in models)
- Random Forest (using buit-in models)

Moreover, I have implemented functions for calculating accuracy, precision, recall, confusion matrix, and F1 socre.

Different preprocessing techniques such as stop-word removal, stemming, and lemmatization are also explored.

For clustering, I have also implemented k-Means, and the results are shown by applying t-SNE on the data.



