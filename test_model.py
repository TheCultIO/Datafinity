import model
import pandas as pd
model.train_model()
result_df = model.predict_model(pd.read_csv('sample.csv'))
print(result_df.head())
print(result_df['predicted_pricePerSquareFoot'])
