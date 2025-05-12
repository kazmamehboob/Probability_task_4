from sklearn.metrics import accuracy_score
import pickle

X_train,X_test,y_train,test_filenames=pickle.load(open("E:\Project\models\split.pkl","rb"))
model=pickle.load(open("E:\Project\models\model.pkl","rb"))


y_pred=model.predict(X_test)
acc=accuracy_score(test_filenames,y_pred)
print(f"Test Accuracy:{acc:.2f}")