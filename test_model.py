import model
import pandas as pd
model.train_model()
result_df = model.predict_model(pd.read_excel('example_data.xlsx'))
result_df.head()
