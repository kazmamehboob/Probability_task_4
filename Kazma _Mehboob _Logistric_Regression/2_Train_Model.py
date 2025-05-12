from sklearn.linear_model import LogisticRegression
import pickle

# Load the dataset split
X_train, X_test, y_train, test_filenames = pickle.load(open(r"E:\Project\models\split.pkl", "rb"))
print(X_train[:5])

# Train the model
model = LogisticRegression(multi_class="ovr", max_iter=1000)  # ovr or multinomial depending on your case
model.fit(X_train, y_train)

# Save the trained model to a separate file
pickle.dump(model, open(r"E:\Project\models\model.pkl", "wb"))
