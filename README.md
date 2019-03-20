# Deep Learning Author Prediction

This is a natural-language processing program I wrote that can predict who wrote a blog article based only on the text of the post's title, headline, and description.

For this project, I scraped a list of blog posts on Vice's "Motherboard" blog (__[www.motherboard.vice.com](http://www.motherboard.com)__).  I already executed the scraping script and have not included it in this Notebook.  I gathered about 15,000 different blog posts spanning 7 years.

### Process overview:
1. Open the scraped data in a Pandas dataframe; shuffle the data.
2. We lemmatize each word in the text features to boost semantic understanding.
3. For each of the three text features ("Title", "Headline", and "Description"), we get a list of the 4000 most common words.  We will use these 4000 words as feature maps for each of the three text features.
4. We populate the data table with the augmented features.
5. We binarize the labels, converting the authors' names into a one-hot encoded matrix.
6. We run the data through a Keras deep neural network several times, with different cross-validation splits.
7. We analyze the results, learning that Validation Accuracy peaks at the third epoch.
8. We create our final model and run the test data through it, and achieve a score of 0.40625.  **41% accuracy!**

I used a relatively naive approach to feature generation; for each sample, I created a feature set of which words in the sample were in a list of the most common words in all the blog posts.  I could absolutely do more feature augmentation to achieve a higher model accuracy.

Sven Zetterlund