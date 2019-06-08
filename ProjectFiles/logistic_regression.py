import csv
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split



def pre_process_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews


reviews_train = []
ratings_train = []
count_train = 0
with open('drugsComTrain_raw.tsv') as train_file:
    readCSV = csv.reader(train_file, delimiter='\t')
    for row in readCSV:
        reviews_train.append(row[3])
        ratings_train.append(row[4])
        count_train += 1

print(ratings_train[1])        
reviews_test = []
ratings_test = []
count_test = 0
with open('drugsComTest_raw.tsv') as test_file:
    readCSV = csv.reader(test_file, delimiter='\t')
    for row in readCSV:
        reviews_test.append(row[3])
        ratings_test.append(row[4])
        count_test += 1

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

reviews_train_clean = pre_process_reviews(reviews_train[1:count_train])
reviews_test_clean = pre_process_reviews(reviews_test[1:count_test])

#print(reviews_train_clean[0])

stop_words = ['in', 'of', 'at', 'a', 'the']

cv = CountVectorizer(binary=True)
# cv = CountVectorizer(binary=True)
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
X_test = cv.transform(reviews_test_clean)

target_train = [1 if float(ratings_train[i]) > 6 else -1 if float(ratings_train[i]) < 5 else 0 for i in range(1,count_train)]
target_test = [1 if float(ratings_test[i]) > 6 else -1 if float(ratings_test[i]) < 5 else 0 for i in range(1,count_test)]

X_train, X_val, y_train, y_val = train_test_split(
    X, target_train, train_size=0.75
)

# for c in [0.01, 0.05, 0.25, 0.5, 1]:
#     lr = LogisticRegression(C=c)
#     lr.fit(X_train, y_train)
#     print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))

final_model = LogisticRegression(C=1)
final_model.fit(X, target_train)
print ("Final Accuracy Score: %s" % accuracy_score(target_test, final_model.predict(X_test)))
print ("F1 Score: %s" % f1_score(target_test, final_model.predict(X_test), average='weighted'))
print ("Precision Score: %s" % precision_score(target_test, final_model.predict(X_test), average='macro'))
print ("Recall Score: %s" % recall_score(target_test, final_model.predict(X_test), average='micro'))
print ("Confusion Matrix: %s" % confusion_matrix(target_test, final_model.predict(X_test)))

feature_to_co_ef = {
    word: co_ef for word, co_ef in zip(
        cv.get_feature_names(), final_model.coef_[0]
    )
}

for best_positive in sorted(
        feature_to_co_ef.items(),
        key=lambda x: x[1],
        reverse=True)[:5]:
    print (best_positive)

for best_negative in sorted(
        feature_to_co_ef.items(),
        key=lambda x: x[1])[:5]:
    print (best_negative)


