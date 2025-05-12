# # save_model.py
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import joblib

# # Example dataset
# data = pd.DataFrame({
#     'age': [25, 32, 47, 51, 62],
#     # 'income': [""" 50000, 60000, 80000, 82000, 90000 """],
#     'loan_amount': [10000, 15000, 20000, 25000, 30000],
#     'approved': [0, 1, 1, 0, 1]
# })

# X = data[['age', 'income', 'loan_amount']]
# y = data['approved']

# # Train the model
# model = RandomForestClassifier()
# model.fit(X, y)

# # Save the model
# joblib.dump(model, 'model.pkl')
# print("✅ Model saved as model.pkl")