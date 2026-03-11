import pickle
with open('sign_language_model.p', 'rb') as f:
    model = pickle.load(f)
print(list(model.classes_))
