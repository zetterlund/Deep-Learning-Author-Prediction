import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.preprocessing import label_binarize, StandardScaler, MinMaxScaler, Normalizer, LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from keras import models
from keras import layers
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('bmh')



# Here we import the CSV of our scraped results.  There are about 15,000 entries.
df = pd.read_csv("files/articles.csv", names=['id', 'url', 'title', 'headline', 'description', 'author', 'date'], skiprows=1)
df.drop(['id', 'url'], axis=1, inplace=True)

# Shuffle the rows for randomness
df = df.sample(frac=1).reset_index(drop=True)

# Convert date text to datetime format
df['date'] = pd.to_datetime(df['date'])

# We filter the results down to only the 20 most popular authors.  This leaves us with about 9,000 entries.
authors = df['author'].value_counts().to_frame()
authors = authors.index[:20].tolist()
df = df[df['author'].isin(authors)]

# Split the data into training and testing sets:
X_train, X_test, y_train, y_test = train_test_split(df.drop('author', axis=1), df['author'], test_size=0.3, random_state=0)



'''Some functions for returning the normalized value of the age of the articles'''
# Get age in weeks since 2000
def get_time_diff(s):
    time_diff = (s - pd.to_datetime('2000-1-1'))
    time_diff = time_diff / np.timedelta64(1, 'W')
    return time_diff

# Fit the scaler to the training data
def fit_age_scaler(X):    
    age = map(get_time_diff, X_train['date'])
    age = np.array(list(age))
    age = age.reshape(-1, 1)
    age_scaler = MinMaxScaler()
    age_scaler.fit(age)    
    return age_scaler
age_scaler = fit_age_scaler(X_train)

# Function for later transforming date data into age
def get_normalized_age(X):
    age = map(get_time_diff, X['date'])
    age = np.array(list(age))
    age = age.reshape(-1, 1)
    normalized_age = age_scaler.transform(age)
    return normalized_age



'''Miscellaneous helper functions'''
# Convert tags to work properly with WordNetLemmatizer
def pos_converter(word):
    pos = nltk.pos_tag([word])[0][1]
    if pos.startswith('J'):
        return wordnet.ADJ
    elif pos.startswith('V'):
        return wordnet.VERB
    elif pos.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
# Set up sentiment dictionary for sentiment analysis
sentiment_dictionary = {}
for line in open('files/AFINN-111.txt'):
    word, score = line.split('\t')
    sentiment_dictionary[word] = int(score)

# Feature words dictionary for separating different text fields
feature_words_dict = {}

# Instantiate Lemmatizer and create list of stop words:
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))



'''For each of the three text fields ('Title', 'Headline', and 'Description'), we get a list of the most common 4000 words across all records'''
for text_column in ['title', 'headline', 'description']:

    # To boost the semantic understanding, we lemmatize (akin to stemming) each word before analyzing it, while maintaining the table structure
    lemmatized_cells = []
    for cell in X_train[text_column]:        
        cell_words = []        
        if len(str(cell)) > 0:
            tokenized_cell = word_tokenize(str(cell))
            for (word, pos) in nltk.pos_tag(tokenized_cell):
                cell_words.append(lemmatizer.lemmatize(word.lower(), pos=pos_converter(pos)))
        lemmatized_cells.append(cell_words)

    # Get list of all words used in the lemmatized cells
    all_words = []
    for cell in lemmatized_cells:
        for word in cell:
            all_words.append(word)

    # Eliminate stop words from the all_words list   
    all_words = [value for value in all_words if value not in stop_words]

    # Create frequency distribution of all words
    all_words = nltk.FreqDist(all_words)

    # Take top 4000 most common words to define the feature sets
    feature_words = all_words.most_common(4000)
    feature_words = list(a for a,b in feature_words)
    
    feature_words_dict[text_column] = feature_words



# Here is a function for properly setting up the features of the data.  We will run this function on both the Training and Testing feature sets:
def feature_setup_pipeline(X):  
    feature_array = []
    
    # Create list of features
    for text_column in feature_words_dict:
        
        text_array = []
        for cell in X[text_column]:

            cell_words = []
            matrix = np.zeros(4000)

            if len(str(cell)) > 0:

                tokenized_cell = word_tokenize(str(cell))
                lemmatized_list = []

                for (word, pos) in nltk.pos_tag(tokenized_cell):
                    lemmatized_list.append(lemmatizer.lemmatize(word.lower(), pos=pos_converter(pos)))

                for i, feature_word in enumerate(feature_words_dict[text_column]):
                    if feature_word in lemmatized_list:
                        matrix[i] = 1

            text_array.append(matrix)        
        feature_array.append(text_array)        
        
    # Add a feature for which "batch" the article is part of (the old one or the new one)
    X['old batch'] = 0
    X['new batch'] = 0
    X.loc[X['date'] > '2015-1-1', 'new batch'] = 1
    X.loc[X['date'] < '2015-1-1', 'old batch'] = 1
    
    # Add a feature for the normalized age of the article
    X['age'] = get_normalized_age(X) 
    
    # Drop unnecessary columns:
    X = X.drop(['title', 'headline', 'description', 'date'], axis=1)  

    # Place the featured word lists back into the data:
    for i in feature_array:
        X = np.concatenate((X, i), axis=1)
        
    return X
   


'''Here is some code to add sentiment analysis to each of the text fields.  This has not yet been implemented.'''
    # # Add 3 different sentiment values to the features:
    # rev_sent_absolute = []
    # rev_sent_adjusted = []
    # rev_sent_valence = []
    # for review in X:
    #     rev_sent_absolute.append(sum(sentiment_dictionary.get(word, 0) for word in review))
    #     rev_sent_adjusted.append(sum(sentiment_dictionary.get(word, 0) for word in review) / len(review))
    #     rev_sent_valence.append(sum(math.fabs(sentiment_dictionary.get(word, 0)) for word in review) / len(review))



# Run the data through the feature setup pipeline:
X_train = feature_setup_pipeline(X_train)
X_test = feature_setup_pipeline(X_test)

# Now we have our feature data all properly formatted.  Let's binarize the categorical labels:
y_train = label_binarize(y_train, classes=authors)
y_test = label_binarize(y_test, classes=authors)



'''Model Training:'''
# Define a function for model instantiation:
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1], )))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(20, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

'''Run the model several times with different cross-validation splits:'''
num_epochs = 10

all_val_acc_histories = []
kf = KFold(n_splits=3, shuffle=True, random_state=0)

for index, (train_indices, val_indices) in enumerate(kf.split(X_train, y_train)):
    print('processing fold #', index)
    
    xtrain, xval = X_train[train_indices], X_train[val_indices]
    ytrain, yval = y_train[train_indices], y_train[val_indices]

    # Build the model
    model = build_model()
    
    # Train the model
    history = model.fit(xtrain, ytrain, validation_data=(xval, yval), epochs=num_epochs, batch_size=32, verbose=0)
    
    # Record model validation metrics
    val_acc_history = history.history['val_acc']
    all_val_acc_histories.append(val_acc_history)



# Get the average accuracy amongst all k-folds for each epoch:
average_val_acc_history = [np.mean([x[i] for x in all_val_acc_histories]) for i in range(num_epochs)]

# Plot the results
plt.figure(figsize=(14, 6))
plt.plot(np.array(range(len(average_val_acc_history))) + 1, average_val_acc_history)
plt.xticks(np.arange(0, 12, 1))
plt.xlim(0,11)
plt.ylim(.25, 0.45)
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.show()

# It looks like the average validation accuracy peaks at epoch 3.  Now let's create our final model.

# Build and train final model
model = build_model()
history = model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=1)



'''Model Testing'''
# Now it's time to put our money where our mouth is - time to test the data.
results = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', results[1]) # 0.40625 - 41% accuracy!