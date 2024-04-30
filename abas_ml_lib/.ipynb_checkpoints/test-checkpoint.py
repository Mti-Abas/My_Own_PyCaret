from __init__ import Read_And_EDA, AutoML
read_eda=Read_And_EDA()
unclean_data=read_eda.readFile('C:\\Users\\Mti.Abas\\Desktop\\مبادرة 10 الاف مبرمج ذكاء اصطناعي\\مشروع تعلم الالة\\loan_data_Copy.csv')
unclean_data.head()
clean_data=read_eda.do_EDA(unclean_data)
# Initialize AutoML and find the best model
automl = AutoML(clean_data, target_column="Loan_Status")
automl.find_best_model()